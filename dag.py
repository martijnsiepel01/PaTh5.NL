import pandas as pd
from transformers import AutoTokenizer

def load_thesaurus(thesaurus_location):
    thesaurus = pd.read_csv(thesaurus_location, sep="|", encoding="latin")
    return thesaurus

def get_unique_codes(thesaurus):
    unique_codes = thesaurus[thesaurus["DESTACE"] == "V"]["DEPALCE"].str.lower().unique().tolist()
    return unique_codes

def divide_codes(unique_codes):
    topography = [c for c in unique_codes if c.startswith("t")]
    procedure = [c for c in unique_codes if c.startswith("p")]
    morphology = [c for c in unique_codes if not (c.startswith("t") or c.startswith("p"))]
    return topography, procedure, morphology

def tokenize_codes(tokenizer, codes):
    tokens_set = set()
    for word in codes:
        tokens = tokenizer.tokenize(word)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokens_set.update(token_ids)
    return tokens_set

def create_tokenized_sets(tokenizer, topography, procedure, morphology):
    topography_tokens = tokenize_codes(tokenizer, topography)
    procedure_tokens = tokenize_codes(tokenizer, procedure)
    morphology_tokens = tokenize_codes(tokenizer, morphology)
    return topography_tokens, procedure_tokens, morphology_tokens

def load_data(data_location):
    df = pd.read_csv(data_location, sep="\t", usecols=["Codes"])
    df["Codes"] = df["Codes"].str.lower().str.split()
    df["Codes"] = df["Codes"].apply(
        lambda codes: [c.replace("[c-sep]", "[C-SEP]") for c in codes] if codes else []
    )
    df.dropna(inplace=True)
    return df

def split_on_token(lst, token="[c-sep]"):
    parts = []
    start = 0
    while True:
        try:
            idx = lst.index(token, start)
            parts.append(lst[start:idx])
            start = idx + 1
        except ValueError:
            parts.append(lst[start:])
            break
    return parts

def split_data_on_token(data, token="[c-sep]"):
    data["Codes"] = data["Codes"].apply(lambda x: split_on_token(x, token))
    return data

def explode_data(data):
    return data.explode("Codes").reset_index(drop=True)

def tokenize_exploded_data(data_exploded, tokenizer):
    def encode_chunk(codes_list):
        if isinstance(codes_list, list):
            txt = " ".join(codes_list)
            return tokenizer.encode(txt, add_special_tokens=False)
        return []
    data_exploded["Encoded_Codes"] = data_exploded["Codes"].apply(encode_chunk)
    return data_exploded

def load_mutually_exclusive_terms(file_path):
    with open(file_path, "r") as f:
        data = [line.strip().lower().split(",") for line in f]
    return pd.DataFrame(data, columns=["Term1","Term2"])

def create_exclusive_dict(mutually_exclusive_terms):
    exclusive_dict = {}
    for _, row in mutually_exclusive_terms.iterrows():
        t1, t2 = int(row["Term1"]), int(row["Term2"])
        exclusive_dict[t1] = t2
        exclusive_dict[t2] = t1
    return exclusive_dict

class DAG:
    def __init__(
        self,
        base_dict=None,
        tokenizer=None,
        exclusive_dict=None,
        topography_tokens=None,
        procedure_tokens=None,
        morphology_tokens=None
    ):
        self.dag_dict = base_dict if base_dict else {}
        self.tokenizer = tokenizer
        self.c_sep_tokens = self.tokenize_c_sep() if tokenizer else []
        self.node_metadata = {}
        self.exclusive_dict = exclusive_dict or {}
        self.topography_tokens = topography_tokens or set()
        self.procedure_tokens = procedure_tokens or set()
        self.morphology_tokens = morphology_tokens or set()

    def tokenize_c_sep(self):
        return self.tokenizer.encode("[C-SEP]", add_special_tokens=False) if self.tokenizer else []

    def add(self, sequence):
        current_dict = self.dag_dict
        current_type = "topography"
        for code in sequence:
            if code not in current_dict:
                current_dict[code] = {}
            if not self.is_valid_transition(current_type, code):
                # Skip or handle differently if invalid
                continue
            current_dict = current_dict[code]

            if code in self.topography_tokens:
                self._add_tokenized_c_sep(current_dict)
                current_type = "topography"
            elif code in self.procedure_tokens:
                current_type = "procedure"
            elif code in self.morphology_tokens:
                self._add_tokenized_c_sep(current_dict)
                current_type = "morphology"

            node_key = id(current_dict)
            self.node_metadata[node_key] = {"type": current_type}

    def is_valid_transition(self, current_type, code):
        if current_type == "topography":
            return code in self.topography_tokens or code in self.procedure_tokens or code in self.morphology_tokens
        elif current_type == "procedure":
            return code in self.procedure_tokens or code in self.morphology_tokens
        elif current_type == "morphology":
            return code in self.morphology_tokens or code in self.topography_tokens
        return True

    def _add_tokenized_c_sep(self, current_dict):
        for token in self.c_sep_tokens:
            if token not in current_dict:
                current_dict[token] = {}
            current_dict = current_dict[token]

    def get(self, prefix_sequence):
        current_dict = self.dag_dict
        for code in prefix_sequence:
            if code in current_dict:
                current_dict = current_dict[code]
            else:
                return []
        possible_continuations = list(current_dict.keys())

        for code in prefix_sequence:
            if code in self.exclusive_dict:
                conflicting = self.exclusive_dict[code]
                possible_continuations = [
                    pc for pc in possible_continuations if pc != conflicting
                ]
        # Avoid repeating codes that already appeared
        possible_continuations = [
            pc for pc in possible_continuations if pc not in prefix_sequence
        ]

        node_key = id(current_dict)
        if node_key in self.node_metadata and self.node_metadata[node_key].get("type") == "morphology":
            possible_continuations.append(self.tokenizer.eos_token_id)

        return possible_continuations

def create_palga_dag(thesaurus_location, tokenizer_location, data_location, exclusive_terms_file_path):
    """
    Build a DAG that encodes valid code sequences, plus optional exclusive-code logic.
    """
    thesaurus = load_thesaurus(thesaurus_location)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_location)
    unique_codes = get_unique_codes(thesaurus)
    topography, procedure, morphology = divide_codes(unique_codes)
    topography_tokens, procedure_tokens, morphology_tokens = create_tokenized_sets(
        tokenizer, topography, procedure, morphology
    )

    df = load_data(data_location)
    df = split_data_on_token(df)
    exploded = explode_data(df)
    exploded = tokenize_exploded_data(exploded, tokenizer)

    mutually_exclusive = load_mutually_exclusive_terms(exclusive_terms_file_path)
    exclusive_dict = create_exclusive_dict(mutually_exclusive)

    palga_dag = DAG(
        tokenizer=tokenizer,
        exclusive_dict=exclusive_dict,
        topography_tokens=topography_tokens,
        procedure_tokens=procedure_tokens,
        morphology_tokens=morphology_tokens
    )
    for _, row in exploded.iterrows():
        codes = row["Encoded_Codes"]
        palga_dag.add(codes)

    return palga_dag
