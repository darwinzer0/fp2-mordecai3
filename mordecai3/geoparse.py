from collections import Counter
import logging
import numpy as np
import os
import spacy
import torch
import re
import warnings

from importlib import resources
try:
    from importlib.resources.abc import Traversable  # type: ignore[import-untyped]
except ImportError:
    # Python < 3.13
    from importlib.abc import Traversable
from torch.utils.data import DataLoader
import jellyfish
import numpy as np
import numpy.typing as npt

from .geonames import GeonamesService, setup_pg_pool
from .mordecai_utilities import spacy_doc_setup
from .torch_model import ProductionData, geoparse_model


logger = logging.getLogger(__name__)


spacy_doc_setup()

def load_nlp():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp

def load_model(model_path, device=None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = geoparse_model(device=device,
                           bert_size=768,
                           num_feature_codes=54) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def guess_in_rel(ent):
    """
    A quick rule-based system to detect common "in" relations, such as 
    "Berlin, Germany" or "Aleppo in Syria".

    It tries to skip series of places and respects sentence boundaries.

    This uses some slightly clunky notation to handle the case in training data
    where we don't have a real span, just tokens. 
    """
    if type(ent) is list:
        ent = ent[0].doc[ent[0].i:ent[-1].i+1]
    try:
        next_ent = [e for e in ent.doc.ents if e.start > ent.end]
    except:
        return ""
    # if it's the last ent in the DOC, assume no "in" relation:
    if not next_ent:
        return ""
    next_ent = next_ent[0]
    # if it's the last ent in the SENT, assume no "in" relation:
    if ent.sent != next_ent.sent:
        return ""
    # If the next entity isn't a place, assume no "in" relation
    if next_ent.label_ not in ['GPE', "LOC", 'EVENT_LOC', 'FAC']:
        return ""
    # there's a following entity, separeted only by "in"
    diff = ent.doc[ent.end:next_ent.start]
    diff_text = [i.text for i in diff]
    if len(diff) <= 2 and "in" in diff_text and "and" not in diff_text:
        return next_ent.text
    # There's a comma relation
    if "," in diff_text:
        # skip if there's a ", and":
        if "and" in diff_text:
            return ""
        # skip if the following ent is followed by a comma
        try:
            if ent.doc[next_ent.end].text in [",", "and"]:
                return ""
        except IndexError:
            logger.warning("Error getting 'next_ent'.")
            return ""
        return next_ent.text
    else:
        return ""


def doc_to_ex_expanded(doc):
    """
    Take in a spaCy doc with a custom ._.tensor attribute on each token and create a list
    of dictionaries with information on each place name entity.

    In the broader pipeline, this is called after nlp() and the results are passed to the 
    Geonames lookup step.

    Parameters
    ---------
    doc: spacy.Doc 
      Needs custom ._.tensor attribute.

    Returns
    -------
    data: list of dicts
    """
    data = []
    doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
    # the "loc_ents" are the ones we use for context. NORPs are useful for context,
    # but we don't want to geoparse them. Anecdotally, FACs aren't so useful for context,
    # but we do want to geoparse them.
    loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'NORP']]
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'FAC']:
            tensor = np.mean(np.vstack([i._.tensor for i in ent]), axis=0)
            other_locs = [i for e in loc_ents for i in e if i not in ent]
            in_rel = guess_in_rel(ent)
            if other_locs:
                locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs if i not in ent]), axis=0)
            else:
                locs_tensor = np.zeros(len(tensor))
            d = {"search_name": ent.text,
                 "tensor": tensor,
                 "doc_tensor": doc_tensor,
                 "locs_tensor": locs_tensor,
                 "sent": ent.sent.text,
                 "in_rel": in_rel,
                "start_char": ent[0].idx,
                "end_char": ent[-1].idx + len(ent[-1].text)}
            data.append(d)
    return data


class Geoparser:

    def __init__(self, 
                 model_path: str | Traversable | None = None,
                 geonames: GeonamesService | None = None,
                 pg_dsn: str | None = None,
                 nlp=None,
                 debug: bool = False,
                 trim=None,
                 device='cpu'):
        """
        Parameters
        ----------
        model_path : str or Traversable or None
            Path to the .pt model weights file. Defaults to the bundled asset.
        geonames : GeonamesService or None
            A pre-constructed GeonamesService instance. If None, one is built
            from pg_dsn.
        pg_dsn : str or None
            libpq connection string, e.g.
            "host=localhost dbname=geoindex user=postgres password=secret"
            Required if geonames is not provided.
        nlp : spaCy Language or None
            Pre-loaded spaCy pipeline. Loaded fresh if None.
        debug : bool
            If True, returns top 4 results per entity rather than the best.
        trim : bool or None
            If True, strips internal ranking keys from output dicts.
        device : str
            'cpu' or 'cuda'.
        """
        if device != "cpu":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.trim = trim

        if not nlp:
            self.nlp = load_nlp()
        else:
            if 'token_tensors' not in nlp.pipe_names:
                try:
                    nlp.add_pipe("token_tensors")
                except Exception as e:
                    logger.info(f"Error loading token_tensors pipe: {e}")
            self.nlp = nlp

        # GeonamesService — accepts a pre-built instance or builds one from DSN
        if geonames is not None:
            self.geonames = geonames
        elif pg_dsn is not None:
            pg_pool = setup_pg_pool(pg_dsn)
            self.geonames = GeonamesService(pg_pool)
        else:
            raise ValueError(
                "Either 'geonames' (a GeonamesService instance) or "
                "'pg_dsn' (a libpq connection string) must be provided."
            )

        if not model_path:
            model_path = resources.files("mordecai3") / "assets/mordecai_2025-08-27.pt"
        self.model = load_model(model_path, device=device)
        self.model.to(device)

    def lookup_city(self, entry):
        """
        Resolve a place entry to its parent city using the geonames_hierarchy
        table (via GeonamesService.get_entry_by_id), replacing the old
        file-based hierarchy.txt dict lookup.
        """
        city_id = ""
        city_name = ""
        if entry['feature_code'] == 'PPLX':
            parent_res = self.geonames.get_entry_by_id(entry['geonameid'])
            if parent_res and parent_res['feature_class'] == "P":
                city_id = parent_res['geonameid']
                city_name = parent_res['name']
            else:
                city_id = entry['geonameid']
                city_name = entry['name']
        elif entry['feature_class'] == 'S':
            parent_res = self.geonames.get_entry_by_id(entry['geonameid'])
            if parent_res and parent_res['feature_class'] == "P":
                city_id = parent_res['geonameid']
                city_name = parent_res['name']
        elif re.search("PPL", entry['feature_code']):
            # all other populated places: return self
            city_name = entry['name']
            city_id = entry['geonameid']
        # anything else: city_id and city_name remain ""
        return city_id, city_name


    def geoparse_doc(self, 
                     text, 
                     debug=False, 
                     trim=True, 
                     known_country=None,
                     max_choices=100):
        """
        Geoparse a document.

        Parameters
        ----------
        text : str or spacy Doc (with ._.tensor attributes)
            The text to geoparse.
        debug : bool
            If True, returns the top 4 results for each geoparsed location,
            rather than the single best. Useful for debugging or annotation.
        trim : bool
            If True (default), removes internal ranking keys from output dicts.
        known_country : str
            If provided, restricts candidates to this ISO3 country code.
        max_choices : int
            Maximum candidate entries per toponym passed to the ranker.

        Returns
        -------
        output : dict
            - "doc_text": input text as string
            - "event_location_raw": EVENT_LOC entity text if present
            - "geolocated_ents": list of geoparsed location dicts
        """
        if type(text) is str:
            doc = self.nlp(text)
        elif type(text) is spacy.tokens.doc.Doc:
            doc = text
        else:
            raise ValueError("Text must be either of type 'str' or 'spacy.tokens.doc.Doc'.")

        doc_ex = doc_to_ex_expanded(doc)
        if doc_ex:
            es_data = add_es_data_doc(doc_ex, self.geonames, max_results=max_choices,
                                      known_country=known_country)

            dataset = ProductionData(es_data, max_choices=max_choices)
            data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                self.model.eval()
                pred_val_list = []
                for input_batch in data_loader:
                    input_batch_on_device = {k: v.to(self.model.device) for k, v in input_batch.items()}
                    pred_val_list.append(self.model(input_batch_on_device))
                pred_val = torch.cat(pred_val_list, dim=0)

        event_doc = doc

        best_list = []
        output = {"doc_text": doc.text,
                 "event_location": '',
                 "geolocated_ents": []}
        if len(doc_ex) == 0:
            return output
        elif len(es_data) == 0:
            return output
        else:
            for (ent, pred) in zip(es_data, pred_val):
                logger.debug("**Place name**: {}".format(ent['search_name']))
                if pred[-1] == pred.max():
                    logger.debug("Model predicts no answer")
                    best = {"search_name": ent['search_name'],
                        "start_char": ent['start_char'],
                        "end_char": ent['end_char']}
                    best_list.append(best)
                    continue

                for n, score in enumerate(pred):
                    if n < len(ent['es_choices']):
                        ent['es_choices'][n]['score'] = score.item()
                results = [e for e in ent['es_choices'] if 'score' in e.keys()]

                if not results:
                    logger.debug("(no results)")
                best = {"search_name": ent['search_name'],
                        "start_char": ent['start_char'],
                        "end_char": ent['end_char']}
                scores = np.array([r['score'] for r in results])
                if len(scores) == 0:
                    logger.debug("No scores found.")
                    continue
                if np.argmax(scores) == len(scores) - 1:
                    logger.debug("Picking final 'null' result.")
                    if len(scores) == 1:
                        logger.debug(f"Only one score found: {results[0]}")
                    if len(scores) > 1:
                        second_best_idx = np.argsort(scores)[-2]
                        second_best = results[second_best_idx]
                        logger.debug(f"Second best result: {second_best.get('name', 'N/A')} (score: {second_best.get('score', 'N/A')})")
                    continue
                results = sorted(results, key=lambda k: -k['score'])
                if results and (not debug):
                    logger.debug("Picking top predicted result")
                    best = results[0]
                    best["search_name"] = ent['search_name']
                    best["start_char"] = ent['start_char']
                    best["end_char"] = ent['end_char']
                    best['city_id'], best['city_name'] = self.lookup_city(best)
                    best_list.append(best)
                if results and debug:
                    logger.debug("Returning top 4 predicted results for each location")
                    best = results[0:4]
                    for b in best:
                        b["search_name"] = ent['search_name']
                        b["start_char"] = ent['start_char']
                        b["end_char"] = ent['end_char']
                        b['city_id'], b['city_name'] = self.lookup_city(b)
                        best_list.append(best)

        if (self.trim or trim) and best_list:
            trim_keys = ['admin1_parent_match', 'country_code_parent_match', 'alt_name_length',
                        'min_dist', 'max_dist', 'avg_dist', 'ascii_dist', 'adm1_count', 'country_count']
            for i in best_list:
                i = [i.pop(key) for key in trim_keys if key in i.keys()]
            output = {"doc_text": doc.text,
                 "event_location_raw": ''.join([i.text_with_ws for i in event_doc.ents if i.label_ == "EVENT_LOC"]).strip(),
                 "geolocated_ents": best_list}
        else:
            output = {"doc_text": doc.text,
                 "event_location_raw": ''.join([i.text_with_ws for i in event_doc.ents if i.label_ == "EVENT_LOC"]).strip(),
                 "geolocated_ents": best_list}
        return output


def add_es_data(ex,
                geonames_service: GeonamesService,
                max_results=50,
                fuzzy=0,
                limit_types=False,
                remove_correct=False,
                known_country=None):
    """
    Run a Geonames/Postgres query for a single example and add the results
    to the object.

    Parameters
    ---------
    ex : dict
        Output of doc_to_ex_expanded.
    geonames_service : GeonamesService
        Postgres-backed GeonamesService instance.
    max_results : int
        Maximum candidates to retrieve.
    fuzzy : int
        0 = exact match. Higher values broaden the search via pg_trgm.
    remove_correct : bool
        If True, removes the correct result (used during training data prep).
    known_country : str or None
        ISO3 country code to restrict candidates.
    """
    max_results = int(max_results)
    fuzzy = int(fuzzy)
    search_name = ex['search_name']

    if 'in_rel' in ex.keys():
        if ex['in_rel']:
            parent_place = geonames_service.get_country_by_name(ex['in_rel'])
            if not parent_place:
                parent_place = geonames_service.get_adm1_country_entry(ex['in_rel'], None)
        else:
            parent_place = None
    else:
        parent_place = None

    search_res = geonames_service.search_by_name(search_name, max_results, fuzzy, limit_types, known_country)
    choices = res_formatter(search_res, search_name, parent_place)

    # Always try a fuzzy search if no results, to avoid empty candidate set.
    if not choices:
        search_res = geonames_service.search_by_name(search_name, max_results, fuzzy+1, limit_types, known_country)
        choices = res_formatter(search_res, ex['search_name'], parent_place)

    if remove_correct:
        choices = [c for c in choices if c['geonameid'] != ex['correct_geonamesid']]

    # Always add a final NULL choice at the end
    logger.debug("Adding NULL choice")
    null_choice = {'feature_code': 'NULL',
            'feature_class': 'NULL',
            'country_code3': 'NULL',
            'lat': 0,
            'lon': 0,
            'name': 'NULL',
            'admin1_code': 'NULL',
            'admin1_name': 'NULL',
            'admin2_code': 'NULL',
            'admin2_name': 'NULL',
            'geonameid': 'NULL',
            'admin1_parent_match': -1,
            'country_code_parent_match': -1,
            'alt_name_length': 0,
            'min_dist': 99.0,
            'max_dist': 99.0,
            'avg_dist': 99.0,
            'ascii_dist': 99.0,
            'adm1_count': 0.0,
            'country_count': 0.0}
    choices.append(null_choice)
    ex['es_choices'] = choices

    if remove_correct:
        ex['correct'] = [False for c in choices]
    else:
        if 'correct_geonamesid' in ex.keys():
            ex['correct'] = [c['geonameid'] == ex['correct_geonamesid'] for c in choices]
    return ex


def add_es_data_doc(doc_ex, conn, max_results=50, fuzzy=0, limit_types=False,
                    remove_correct=False, known_country=None):
    doc_es = []
    for ex in doc_ex:
        with warnings.catch_warnings():
            try:
                es = add_es_data(ex, conn, max_results, fuzzy, limit_types, remove_correct, known_country)
                doc_es.append(es)
            except Warning:
                continue
    if not doc_es:
        return []
    admin1_count = make_admin1_counts(doc_es)
    country_count = make_country_counts(doc_es)

    for i in doc_es:
        for e in i['es_choices']:
            e['adm1_count'] = admin1_count[e['admin1_name']]
            e['country_count'] = country_count[e['country_code3']]
    return doc_es


def res_formatter(res, search_name, parent=None):
    """
    Format Geonames/Postgres results into a form for the ML model, including
    edit distance statistics and parent match features.

    Parameters
    ----------
    res : dict
        Search result in the envelope shape:
        {'hits': {'hits': [{'_source': {...}}, ...]}}
    search_name : str
        The original search term from the document.
    parent : dict or None
        Geonames entry for the inferred parent location (from in_rel).

    Returns
    -------
    choices : list of dicts
    """
    choices = []
    alt_lengths = []
    min_dist = []
    max_dist = []
    avg_dist = []
    ascii_dist = []

    for hit in res['hits']['hits']:
        # Our Postgres envelope uses plain dicts; no .to_dict() call needed.
        i = hit['_source']
        names = [i['name']] + i['alternativenames']
        dists = [jellyfish.levenshtein_distance(search_name, j) for j in names]
        d = {"feature_code": i['feature_code'],
            "feature_class": i['feature_class'],
            "country_code3": i['country_code3'],
            "lat": float(i['lat']),
            "lon": float(i['lon']),
            "name": i['name'],
            "admin1_code": i['admin1_code'],
            "admin1_name": i['admin1_name'],
            "admin2_code": i['admin2_code'],
            "admin2_name": i['admin2_name'],
            "geonameid": i['geonameid']}

        if parent:
            if parent['admin1_name'] == "":
                d['admin1_parent_match'] = 0
            elif parent['admin1_name'] == i['admin1_name']:
                d['admin1_parent_match'] = 1
            else:
                d['admin1_parent_match'] = -1

            if parent['country_code3'] == "":
                d['country_code_parent_match'] = 0
            elif parent['country_code3'] == i['country_code3']:
                d['country_code_parent_match'] = 1
            else:
                d['country_code_parent_match'] = -1
        else:
            d['admin1_parent_match'] = 0
            d['country_code_parent_match'] = 0

        choices.append(d)
        alt_lengths.append(len(i['alternativenames']) + 1)
        min_dist.append(np.min(dists))
        max_dist.append(np.max(dists))
        avg_dist.append(np.mean(dists))
        ascii_dist.append(jellyfish.levenshtein_distance(search_name, i['asciiname']))

    alt_lengths = np.log(alt_lengths)
    min_dist = normalize(min_dist)
    max_dist = normalize(max_dist)
    avg_dist = normalize(avg_dist)
    ascii_dist = normalize(ascii_dist)

    for n, i in enumerate(choices):
        i['alt_name_length'] = alt_lengths[n]
        i['min_dist'] = min_dist[n]
        i['max_dist'] = max_dist[n]
        i['avg_dist'] = avg_dist[n]
        i['ascii_dist'] = ascii_dist[n]
    return choices


def make_admin1_counts(out):
    """
    Get the ADM1s from all candidate results for all locations in a document
    and return the proportion of place names that have at least one candidate
    from each ADM1. Used as a document-level coherence feature.
    """
    admin1s = []
    for es in out:
        other_adm1 = set([i['admin1_name'] for i in es['es_choices']])
        admin1s.extend(list(other_adm1))
    admin1_count = dict(Counter(admin1s))
    for k, v in admin1_count.items():
        admin1_count[k] = v / len(out)
    return admin1_count


def make_country_counts(out):
    """
    Get the countries from all candidate results for all locations in a
    document and return the proportion of place names with candidates from
    each country. Used as a document-level coherence feature.
    """
    all_countries = []
    for es in out:
        countries = set([i['country_code3'] for i in es['es_choices']])
        all_countries.extend(list(countries))
    country_count = dict(Counter(all_countries))
    for k, v in country_count.items():
        country_count[k] = v / len(out)
    return country_count


def normalize(ll: list[float]) -> npt.NDArray[np.float64]:
    """Normalize an array to [0, 1]"""
    arr = np.array(ll)
    if len(arr) > 0:
        max_arr = np.max(arr)
        if max_arr == 0:
            max_arr = 0.001
        arr = (arr - np.min(arr)) / max_arr
    return arr
