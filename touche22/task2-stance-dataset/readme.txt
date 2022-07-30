The data sources are the Stack Exchange dumps [https://archive.org/details/stackexchange] and L6 - Yahoo! Answers Comprehensive Questions and Answers version 1.0 (multi part) [https://webscope.sandbox.yahoo.com/catalog.php?datatype=l]

Since the Yahoo! Answers data is bound by the non-disclose agreement, we provide the question and answer ids for about a half of the dataset entries. We ask you to obtain the dataset from webscope.sandbox.yahoo.com directly and use the notebook process_stance_dataset.ipynb to fetch the questions and answers for the Yahoo part.
 
The dataset structure:

ds: the data source
id: entry id (not used)
question: question from Yahoo or Stack Exchange
answer: "best" / "accepted" answer from Yahoo or Stack Exchange
answer_stance: 0: No stance, 1: Neutral, 2: Pro first object, 3: Pro second object
answer_stance object: The object with the pro stance is named
object_1, object_2: First / second compared object in the question
mask_pos_1, mask_pos_2: The list of position of the object_1 / object_2 mentions in the answer (Note: the objects were labeled manually, so that the objects in the answer can be syntactically different from the ones in the question, e.g., synonyms, acronyms, abbreviations, etc.)

For the dataset citation, please use the following bib-entry:

@inproceedings{bondarenko:2022a,
  author    = {Alexander Bondarenko and Yamen Ajjour and Valentin Dittmar and Niklas Homann and Pavel Braslavski and Matthias Hagen},
  title     = {{Towards Understanding and Answering Comparative Questions}},
  booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining (WSDM 2022)},
  publisher = {{ACM}},
  year      = {2022},
  doi       = {10.1145/3488560.3498534}
}
