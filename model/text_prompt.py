import torch
from clip.clip import tokenize


def text_prompt(name_list: list):
    # text_aug = [f"a dark photo of action {{}}", f"a silhouette picture of action {{}}", f"Human action of {{}} in darkness", f"{{}}, an action",
    #             f"{{}} this is an action in dark environment", f"{{}}, a low brightness video of action", f"Playing action of {{}} at night", f"{{}}",
    #             f"Contour of playing a kind of action barely seen, {{}}", f"Doing a kind of action, low light, {{}}", f"Look carefully, the human is {{}} in the dark",
    #             f"Can you recognize the silhouette of action of {{}} step by step?", f"Video contour classification of {{}}", f"A silhouette video shot at night of {{}}",
    #             f"The man is {{}} in the dark", f"The woman is {{}} in dim light"]
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    num_text_aug = len(text_aug)

    text_tokenized = torch.stack([torch.stack(
        [tokenize(txt.format(name)) for name in name_list]) for txt in text_aug]).squeeze()

    #classes = torch.cat([v for k, v in text_mat.items()])

    return num_text_aug, text_tokenized
