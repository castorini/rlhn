import logging
import re
import random

random.seed(42)

logger = logging.getLogger(__name__)

class RLHNTechnique:
    def __init__(self, doc_regex: str = r"Doc \((\d+)\)", threshold: int= 8):
        self.doc_regex = doc_regex
        self.threshold = threshold

    @staticmethod
    def _cleanup(data):
        """
        Cleanup the data
        """
        data.pop("positive_passage_ids", None)
        data.pop("negative_passage_ids", None)
        data.pop("response", None)
        data.pop("finish_reason", None)
        return data

    def _postprocess(
        self,
        response: str,
        filter_string: str) -> list[str]:
        try:
            matches_better, matches_worse = [], []
            verdict = response.split("<verdict>")[1].split("</verdict>")[0]
            if filter_string in ["better", "both"]:
                better = verdict.split("<better>")[1].split("</better>")[0].strip()
                matches_better = re.findall(self.doc_regex, better)
            if filter_string in ["worse", "both"]:
                worse = verdict.split("<worse>")[1].split("</worse>")[0].strip()
                matches_worse = re.findall(self.doc_regex, worse)

            # find all doc_ids where false negatives are either better or worse
            matches = matches_better + matches_worse
            return matches

        except Exception as e:
            logger.error(f"Error in processing. Error: {e}")
            return []

    def default(self, data):
        """
        Default technique to remove the fields
        """
        self._cleanup(data)
        return data

    def remove(self, data, filter_string: str):
        """
        Remove the whole datapoint from the training dataset
        """
        response = data["response"]
        matches = self._postprocess(response, filter_string)

        if len(matches) == 0:
            # if no matches found, means all negatives are correctly labeled
            self._cleanup(data)
            return data

    def hn_remove(self, data, filter_string: str):
        """
        Remove only the hard negatives from the hard negative subset
        """
        response = data["response"]
        matches = self._postprocess(response, filter_string)
        indices = [int(match) - 1 for match in matches]
        remove_doc_ids = [data["negative_passage_ids"][idx] for idx in indices]

        if len(indices) <= self.threshold:
            positive_passages = data["positive_passages"]
            negative_passages = data["negative_passages"]

            # remove the doc_ids from the negative passages
            negative_passages = [
                passage for passage in negative_passages if passage["docid"] not in remove_doc_ids
            ]
            data["positive_passages"] = positive_passages
            data["negative_passages"] = negative_passages
            self._cleanup(data)
            return data
        else:
            # if more than threshold matches found, remove the whole datapoint
            return None

    def rlhn(self, data, filter_string: str):
        """
        RLHN technique to replace the false negatives with the positive passages
        """
        response = data["response"]
        matches = self._postprocess(response, filter_string)
        indices = [int(match) - 1 for match in matches]
        remove_doc_ids = [data["negative_passage_ids"][idx] for idx in indices]

        if len(indices) <= self.threshold:
            positive_passages = data["positive_passages"]
            negative_passages = data["negative_passages"]

            # replace the doc_ids from the negative passages with the positive passages
            for idx in indices:
                positive_passages.append(negative_passages[idx])

            # remove the doc_ids from the negative passages
            negative_passages = [
                passage for passage in negative_passages if passage["docid"] not in remove_doc_ids
            ]
            data["positive_passages"] = positive_passages
            data["negative_passages"] = negative_passages
            self._cleanup(data)
            return data
        else:
            # if more than threshold matches found, remove the whole datapoint
            return None

    def modify(self, data, technique: str, filter_string: str):
        """"""
        if technique not in [
            "default",
            "remove",
            "hn_remove",
            "rlhn",
        ]:
            raise ValueError(f"Technique {technique} not supported")
        else:
            if technique == "default":
                return self.default(data)
            elif technique == "remove":
                return self.remove(data, filter_string)
            elif technique == "hn_remove":
                return self.hn_remove(data, filter_string)
            elif technique == "rlhn":
                return self.rlhn(data, filter_string)

    def identify(self, data, filter_string: str):
        """
        Identify whether atleast one false negative present or not
        """
        response = data["response"]
        matches = self._postprocess(response, filter_string)

        if len(matches) == 0:
            # if no matches found, means all negatives are correctly labeled
            return False
        else:
            # if matches found, means atleast one false negative present
            return True

