import torch
from clip.clip import tokenize


def text_prompt(name_list: list):
    # text_aug = [f"a dark photo of action {{}}", f"a picture of action {{}} at night", f"Human action of {{}} in darkness", f"{{}}, an action in the darkness",
    #             f"{{}} this is an action in dark environment", f"{{}}, a low brightness video of action", f"Playing action of {{}} at night", f"{{}}",
    #             f"Playing a kind of action barely seen in the dark, {{}}", f"Doing a kind of action, low light, {{}}", f"Look carefully, the human is {{}} in the dark",
    #             f"Can you recognize the silhouette of action of {{}} step by step?", f"Video classification of {{}}", f"A video shot at night of {{}}",
    #             f"The man is {{}} in the dark", f"The woman is {{}} in dim light"]
    text_aug = [f"a photo of action {{}} at night", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}} in the dim area", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look carefully, the human is {{}} in the darkness",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}} at night",
                f"The man is {{}} in the dark area", f"The woman is {{}}"]
    num_text_aug = len(text_aug)

    text_tokenized = torch.stack([torch.stack(
        [tokenize(txt.format(name)) for name in name_list]) for txt in text_aug]).squeeze()

    #classes = torch.cat([v for k, v in text_mat.items()])

    return num_text_aug, text_tokenized
