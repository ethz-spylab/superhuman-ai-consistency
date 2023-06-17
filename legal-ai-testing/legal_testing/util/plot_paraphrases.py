from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from highlight_text import HighlightText
import matplotlib.axes


def auto_wrap_sentence(sentence: str, line_length_max: int) -> str:
    """
    Automatically wrap a sentence to a maximum line length.
    """
    words = sentence.split(" ")
    words_clean = [w.replace("\u0336", "") for w in words]
    words_clean = [w.replace("<", "") for w in words_clean]
    words_clean = [w.replace(">", "") for w in words_clean]
    lines_clean = []
    lines = []
    line = ""
    line_clean = ""
    for index, word in enumerate(words_clean):
        if len(line_clean) + len(word) > line_length_max:
            lines_clean.append(line_clean[:-1])
            lines.append(line[:-1])
            line_clean = ""
            line = ""
        line_clean += word + " "
        line += words[index] + " "
    lines_clean.append(line_clean[:-1])
    lines.append(line[:-1])
    return "\n".join(lines)


def get_levenshtein_sentence_distance_edits(string1, string2) -> List[str]:
    # Only consider additions and deletions
    distance_dict: Dict[Tuple(int, int), int] = {}

    sentence1 = string1.split(" ")
    sentence2 = string2.split(" ")

    # Add the base cases
    for i in range(len(sentence1) + 1):
        distance_dict[(i, 0)] = i

    for j in range(len(sentence2) + 1):
        distance_dict[(0, j)] = j

    # Fill the rest of the table
    for i in range(1, len(sentence1) + 1):
        for j in range(1, len(sentence2) + 1):
            if sentence1[i - 1] == sentence2[j - 1]:
                distance_dict[(i, j)] = distance_dict[(i - 1, j - 1)]
            else:
                distance_dict[(i, j)] = min(
                    distance_dict[(i - 1, j)] + 1,
                    distance_dict[(i, j - 1)] + 1,
                )

    # Get a list of the edits
    edits = []
    i, j = len(sentence1), len(sentence2)
    while i > 0 or j > 0:
        if i == 0:
            edits.append("+")
            j -= 1
        elif j == 0:
            edits.append("-")
            i -= 1
        elif sentence1[i - 1] == sentence2[j - 1]:
            edits.append("0")
            i -= 1
            j -= 1
        elif distance_dict[(i - 1, j)] <= distance_dict[(i, j - 1)]:
            edits.append("-")
            i -= 1
        else:
            edits.append("+")
            j -= 1

    return edits[::-1]


def create_text_object(
    sentence: str, bboxes: List[Dict[str, Any]], ax: matplotlib.axes.Axes, **kwargs
):
    custom_gray = "#969696"
    default_args = {
        "x": 0.5,
        "y": 0.5,
        "fontsize": 16,
        "ha": "center",
        "va": "center",
        "s": sentence,
        "highlight_textprops": bboxes,
        "annotationbbox_kw": {
            "frameon": True,
            "pad": 2,
            "bboxprops": {"facecolor": "white", "edgecolor": custom_gray, "linewidth": 5},
        },
        "ax": ax,
    }
    arguments = {**default_args, **kwargs}
    HighlightText(**arguments)


def plot_paraphrases(
    sentence1: str,
    sentence2: str,
    save_plot: bool = False,
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    if save_plot and save_path is None:
        raise ValueError("If save_plot is True, save_path must be provided.")

    # Get the edit history required to transform sentence1 into sentence2
    edits = get_levenshtein_sentence_distance_edits(sentence1, sentence2)

    sentence1_parts = sentence1.split(" ")
    sentence2_parts = sentence2.split(" ")

    # Build a new sentence by adding deletions with strike-through font and additions with underline font
    new_sentence = []
    bboxes = []
    custom_green = "#a3dc8b"
    custom_red = "#f86060"
    insertion_bbox = {
        "bbox": {"edgecolor": custom_green, "facecolor": custom_green, "linewidth": 1.5, "pad": 1}
    }
    deletion_bbox = {
        "bbox": {"edgecolor": custom_red, "facecolor": custom_red, "linewidth": 1.5, "pad": 1}
    }
    i, j = 0, 0
    for edit in edits:
        if edit == "0":
            new_sentence += [sentence1_parts[i]]
            i += 1
            j += 1
        elif edit == "-":
            word_strike_through = "".join(l + "\u0336" for l in sentence1_parts[i])
            new_sentence += [f"<{word_strike_through}>"]
            bboxes.append(dict(deletion_bbox))
            i += 1
        else:
            new_sentence += [f"<{sentence2_parts[j]}>"]
            bboxes.append(dict(insertion_bbox))
            j += 1

    new_sentence = " ".join(new_sentence)
    formatted_sentence = auto_wrap_sentence(sentence1, line_length_max=60)
    formatted_new_sentence = auto_wrap_sentence(new_sentence, line_length_max=45)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis("off")

    # Create the original sentence text object
    create_text_object(formatted_sentence, [], ax, x=0.25, y=0.5, fontsize=22)

    # Create the new sentence text object
    if save_plot:
        x_pos_second_plot = 4.5
    else:
        x_pos_second_plot = 0.75
    create_text_object(formatted_new_sentence, bboxes, ax, x=x_pos_second_plot, y=0.5, fontsize=22)

    plt.tight_layout()

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close()


if __name__ == "__main__":
    # fmt: off
    # sentence1 = "24. Without communicating it to the applicants, the Directorate presented its report (dated 19 January 2007 and containing its conclusions) to the Commission – the CONSOB proper –, that is, to the body responsible for deciding on possible penalties. At the relevant time the Commission was made up of a chairman and four members, appointed by the President of the Republic on a proposal (su proposta) from the President of the Council of Ministers. Their term of office was for five years and could be renewed only once." # noqa
    # sentence2 = "24. Without communicating it to the applicants, the Directorate presented its report (dated 19 January 2007 and containing its conclusions) to the Commission – the CONSOB proper –, that is, to the body responsible for deciding on possible penalties. At the relevant time the Commission was made up of a chairman and four members, appointed by the President of the Republic on a proposal (su proposta) from the President of the Council of Ministers. Their term was for five years and could only be renewed once." # noqa
    # sentence1 = "5. The applicant complained of the inadequate conditions of detention and of the lack of any effective remedy in domestic law." # noqa
    # sentence2 = "5. The applicant complained about the inadequate conditions of detention and the lack of effective remedy in domestic law." # noqa
    # sentence1 = "8. In November 2006 the applicant suffered a stroke in Russia. Apparently her right side was then paralysed. At the time, she lived with her husband, until he died in 2007. Thereafter the applicant apparently lived with her granddaughter and her family near Vyborg." # noqa
    # sentence2 = "8. In November 2006 the applicant suffered a stroke in Russia. Her right side was apparently paralysed. At the time, she lived with her husband, until he died in 2007. The applicant apparently lived there afterwards with her granddaughter and family near Vyborg." # noqa
    # sentence1 = "24. Without communicating it to the applicants, the Directorate presented its report (dated 19 January 2007 and containing its conclusions) to the Commission – the CONSOB proper –, that is, to the body responsible for deciding on possible penalties. At the relevant time the Commission was made up of a chairman and four members, appointed by the President of the Republic on a proposal (su proposta) from the President of the Council of Ministers. Their term of office was for five years and could be renewed only once." # noqa
    # sentence2 = "24. Without communicating it to the applicants, the Directorate presented its report (dated 19 January 2007 and containing its conclusions) to the Commission – the CONSOB proper –, that is, to the body responsible for deciding on possible penalties. At the relevant time the Commission was made up of a chairman and four members, appointed by the President of the Republic on a proposal (su proposta) from the President of the Council of Ministers. Their term was for five years and could only be renewed once." # noqa
    # sentence1 = "4. Mr L. Dayanov, Ms E. Dayanova, Mr R. Abkadyrov and Mr T. Ulyamayev complained about the excessive length of their pre-trial detention. Mr R. Dayanov complained about his unlawful detention on 7 May 2008." # noqa
    # sentence2 = "4. Mr L. Dayanov, Ms E. Dayanova, Mr R. Abkadyrov and Mr T. Ulyamayev complained of the excessive length of their pre-trial detention. Mr R. Dayanov complained about his unlawful detention on 7 May 2008." # noqa
    # sentence1 = "10. The second, more detailed, hearing took place at the office of the State Secretariat for Migration in Berne on 11 March 2015. A member of a non-governmental organisation was present as a neutral witness, in order to guarantee the fairness of the hearing. He had the opportunity to add comments at the end of the record of the hearing in the event that he had witnessed any irregularities, but did not note down any such observations." # noqa
    # sentence2 = "10. The second, more detailed, hearing was held on 11 march 2015 at the State Secretariat for Migration in Berne. A member of a non-governmental organisation was present as a neutral witness, in order to guarantee the fairness of the hearing. He had the opportunity to add comments at the end of the record of the hearing in the event that he had witnessed any irregularities, but did not note down any such observations." # noqa
    # sentence1 = "9. On 12 May 1994 the Shijak Commission recognised the applicants’ inherited title to a plot of land measuring 5,527 sq. m of which 5,000 sq. m were restored. Since the remaining plot of land measuring 527 sq. m was occupied, the applicants would be compensated in one of the ways provided for by law." # noqa
    # sentence2 = "9. On 12 May 1994 the Shijak Commission recognised the applicants’ inherited title to a plot of land measuring 5,527 sq. m of which 5,000 sq. m were restored. The applicants would be compensated in one of the ways provided for by law since the remaining parcel of land was occupied." # noqa
    # sentence1 = "6. On 25 May 2005 the applicant moved to the city of Rotterdam. She took up residence in rented property at the address A. Street 6b. This address is located in the Tarwewijk district in South Rotterdam. The applicant had previously resided outside the Rotterdam Metropolitan Region (Stadsregio Rotterdam)." # noqa
    # sentence2 = "6. On 25 May 2005 the applicant moved to the city of Rotterdam. She took residence in a rented property at A. Street 6b. This address is located in the Tarwewijk district in South Rotterdam. The applicant had previously resided outside of the Stadsregio Rotterdam metropolitan region." # noqa

    # sentence1 = "16. Following the termination of his military service, on 27 November 2007 Mr Boldyrev applied to the Federal Migration Service for a travel passport. He also submitted medical certificates that attested to the poor health of his parents and justified his need to go and see them."
    # sentence2 = "16. On 27 November 2007, Mr Boldyrev sought a travel passport from the Federal Migration Service after he got out of the military. He backed his request with medical documents that demonstrated his parents' poor health and explained his intention to visit them."

    # sentence1 = "28. In its judgment of 12 October 2011, by a majority of six votes to one, the Supreme Court decided to expel the applicant with a life-long ban on his return."
    # sentence2 = "28. On 12th October 2011, the applicant was expelled from the court by a majority of six votes to one, and was also given a lifetime ban from returning."

    # sentence1 = "9. On 20 November 2012, the IVIMA Housing Inspectorate notified the applicant that the execution of the expulsion was scheduled for 13 December 2012 at 10.00."
    # sentence2 = "9. On November 20, 2012, the IVIMA Housing Inspectorate notified the claimant that the eviction would be executed at 10:00 am on December 13, 2012."

    sentence1 = "41. On 11 January 2012 the applicant was convicted for having possessed a mobile phone while in prison. He was sentenced to imprisonment for seven days."
    sentence2 = "41. The applicant was found guilty of possessing a mobile phone while in prison, and on 11 January 2012, he was convicted and sentenced to seven days in prison."

    # fmt: on

    plot_paraphrases(sentence1, sentence2)
