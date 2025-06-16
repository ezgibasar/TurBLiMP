# TurBLiMP

TurBLiMP is the first Turkish benchmark of linguistic minimal pairs, designed to evaluate the linguistic abilities of monolingual and multilingual language models (LMs). This benchmark covers 16 core grammatical phenomena in Turkish, with 1,000 minimal pairs per phenomenon. Additionally, it incorporates experimental paradigms that examine model performance across different subordination strategies and word order variations.

## Table of Contents
- [:file_folder: Linguistic Phenomena](#file_folder-linguistic-phenomena)
- [:mag_right: Experimental Paradigms](#mag_right-experimental-paradigms)
- [:raising_hand: Human Judgments](#raising_hand-human-judgments)
- [:computer: Usage](#computer-usage)
- [:link: Attribution](#link-attribution)
- [:unlock: License](#unlock-license)

## :file_folder: Linguistic Phenomena

TurBLiMP features 16 core phenomena:

1. **Anaphor Agreement** - Reflexive pronoun agreement violations
2. **Argument Structure (Transitive)** - Case marking errors with transitive verbs
3. **Argument Structure (Ditransitive)** - Case marking errors with ditransitive verbs
4. **Binding** - Principle B violations in binding theory
5. **Determiners** - Obligatory use of the indefinite article
6. **Ellipsis** - Backward gapping with non-parallel word orders
7. **Irregular Forms** - Incorrect aorist allomorph usage
8. **Island Effects** - Wh-adjunct extraction from complex NPs
9. **Nominalization** - Incorrect nominalization suffix selection
10. **NPI Licensing** - Negative polarity items in non-negative contexts
11. **Passives** - Unlicensed use of by-phrases in impersonal passives
12. **Quantifiers** - Quantifier usage with bare nouns
13. **Relative Clauses** - Incorrect case marking in relative clauses
14. **Scrambling** - Illicit postverbal scrambling from embedded clauses
15. **Subject Agreement** - Person/number agreement violations
16. **Suspended Affixation** - Improper tense suffix suspension


## :mag_right: Experimental Paradigms
We also include 20 experimental paradigms targeting the ditransitive and transitive Argument Structure phenomena:

1. **Word Order**  - SOV, SVO, OSV, OVS, VSO, VOS
2. **Subordination** - Finite, -DIK, -(y)IncA, -(y)ken

## :raising_hand: Human Judgments

To validate our benchmark, we collected acceptability judgments from native speakers.

- **Participants**: 30 native Turkish speakers  
- **Rating scale**: 7-point Likert (1: *completely unacceptable* – 7: *completely acceptable*)  
- **Stimuli**:  
  - 216 sentences total  
  - Covers **16 linguistic phenomena** and **20 experimental paradigms**  
  - 6 sentences per category  
- **Design**:  
  - Online survey
  - Two survey versions with flipped acceptability conditions  
- **Data included**:  
  - Raw ratings for all phenomena
  - Survey materials 

## :computer: Usage

To use the TurBLiMP benchmark:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TurBLiMP.git
```

2. For the augmentation module, [the zemberek-full.jar file](https://drive.google.com/file/d/1RRuFK43JqcHcthB3fV2IEpPftWoeoHAu/view?usp=drive_link) should also be downloaded to the project directory.

The dataset is also available on [Hugging Face](https://huggingface.co/datasets/ezgibasar/turblimp)🤗

## :link: Attribution

The citation for the paper associated with this benchmark will be available shortly. If you use the benchmark, please credit the original authors.

| Title                                                     	| Authors                                                       	| Affiliation                                                       	|
|-----------------------------------------------------------	|---------------------------------------------------------------	|-------------------------------------------------------------------	|
| TurBLiMP: A Turkish Benchmark of Linguistic Minimal Pairs 	| Ezgi Başar, Francesca Padovani, Jaap Jumelet, Arianna Bisazza 	| Center for Language and Cognition (CLCG), University of Groningen 	|

## :unlock: License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
