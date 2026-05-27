import torch

reserved_token_mapping = {
    "<|soi|>": 126084,
    "<|eoi|>": 126085,
    "<|sov|>": 126086,
    "<|eov|>": 126087,
    "<|t2i|>": 126088,
    "<|mmu|>": 126089,
    "<|t2v|>": 126090,
    "<|v2v|>": 126091,
    "<|lvg|>": 126092,
    "[iPAD]": 126093,
    "<|r2i|>": 126094,
    "<|act|>": 126095,
    "<|state|>": 126096,
    "<|mm2a|>": 126097,
    "<|soa|>": 126098,
    "<|eoa|>": 126099,
    "<|7dim|>": 126100,
    "<|14dim|>": 126101,
    "<|sostate|>": 126102,
    "<|eostate|>": 126103,
}


class UniversalPrompting:
    def __init__(
        self,
        text_tokenizer,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
            "<|act|>",
            "<|state|>",
            "<|soa|>",
            "<|eoa|>",
            "<|7dim|>",
            "<|14dim|>",
            "<|sostate|>",
            "<|eostate|>",
        ),
        max_text_len=8000,
        max_action_prompt_len=768,
        ignore_id=-100,
        cond_dropout_prob=0.1,
        use_reserved_token=False,
        action_vocab_size=1024,
    ):
        """
        :param text_tokenizer: original text tokenizer
        """
        if not use_reserved_token:
            self.text_tokenizer = text_tokenizer
            self.text_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.text_tokenizer.add_tokens(list(special_tokens))
            self.sptids_dict = {
                token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token]))
                for token in special_tokens
            }
            self.sptids_dict["<|sot|>"] = torch.tensor(
                [self.text_tokenizer.bos_token_id]
            )
            self.sptids_dict["<|eot|>"] = torch.tensor(
                [self.text_tokenizer.eos_token_id]
            )
            self.sptids_dict["<|pad|>"] = torch.tensor(
                [self.text_tokenizer.pad_token_id]
            )
        else:
            self.text_tokenizer = text_tokenizer
            self.sptids_dict = {}
            for token, token_id in reserved_token_mapping.items():
                self.sptids_dict[token] = torch.tensor([token_id])
            self.sptids_dict["<|sot|>"] = torch.tensor(
                [self.text_tokenizer.bos_token_id]
            )
            self.sptids_dict["<|eot|>"] = torch.tensor(
                [self.text_tokenizer.eos_token_id]
            )
            end_header_tokens = self.text_tokenizer.convert_tokens_to_ids(
                ["<|end_header_id|>"]
            )
            if (
                end_header_tokens
                and len(end_header_tokens) > 0
                and end_header_tokens[0]
            ):
                self.sptids_dict["<|end_header_id|>"] = torch.tensor(end_header_tokens)
                self.sptids_dict["<|eot_id|>"] = torch.tensor(
                    self.text_tokenizer.convert_tokens_to_ids(["<|eot_id|>"])
                )
                self.sptids_dict["<|start_header_id|>"] = torch.tensor(
                    self.text_tokenizer.convert_tokens_to_ids(["<|start_header_id|>"])
                )
            else:
                special_tokens_dict = {
                    "additional_special_tokens": [
                        "<|start_header_id|>",
                        "<|end_header_id|>",
                        "<|eot_id|>",
                    ]
                }
                num_added = self.text_tokenizer.add_special_tokens(special_tokens_dict)
                new_token_id = self.text_tokenizer.convert_tokens_to_ids(
                    ["<|end_header_id|>"]
                )
                self.sptids_dict["<|end_header_id|>"] = torch.tensor(new_token_id)
                self.sptids_dict["<|eot_id|>"] = torch.tensor(
                    self.text_tokenizer.convert_tokens_to_ids(["<|eot_id|>"])
                )
                self.sptids_dict["<|start_header_id|>"] = torch.tensor(
                    self.text_tokenizer.convert_tokens_to_ids(["<|start_header_id|>"])
                )
        # plus 1 because at this time we add a task token before
        print(f"self.sptids_dict: {self.sptids_dict}")
        self.action_vocab_size = action_vocab_size
        self.max_text_len = max_text_len + 1  # original mmada prompt setting
        self.max_action_prompt_len = max_action_prompt_len
        self.pad_id = reserved_token_mapping["[iPAD]"]
        self.ignore_id = ignore_id
        self.cond_dropout_prob = cond_dropout_prob

    def t2i_prompt(self, text_ids, image_ids, labels):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = (
                [int(self.sptids_dict["<|t2i|>"])]
                + text_ids[i]
                + [self.text_tokenizer.eos_token_id]
            )

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [
                    int(self.sptids_dict["<|t2i|>"]),
                    self.text_tokenizer.bos_token_id,
                    self.text_tokenizer.eos_token_id,
                ]

            if self.max_text_len >= len(temp_ids):
                old_len = len(temp_ids)
                temp_ids = [self.pad_id] * (
                    self.max_text_len - len(temp_ids)
                ) + temp_ids
                temp_masks = [0] * (self.max_text_len - old_len) + [1] * (
                    old_len + image_ids.shape[-1] + 2
                )
            else:
                # should add the eos token
                temp_ids = temp_ids[: self.max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
                temp_masks = [1] * (
                    len(temp_ids) + image_ids.shape[-1] + 2
                )  # +2 for two special tokens
            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat(
                [
                    # should we predict text tokens when doing image reconstruction?
                    torch.tensor(temp_ids).to(device),
                    self.sptids_dict["<|soi|>"].to(device),
                    labels[i],
                    self.sptids_dict["<|eoi|>"].to(device),
                ],
                dim=0,
            )

            temp_label_ids = torch.where(
                temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids
            )

            temp_ids = torch.cat(
                [
                    torch.tensor(temp_ids).to(device),
                    self.sptids_dict["<|soi|>"].to(device),
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"].to(device),
                ],
                dim=0,
            )

            # sequence_ids: [pad]...[pad] <|t2i|> <bos> text_1 ... text_n <eos> <|soi|> image_1 ... image_m <|eoi|>
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(attention_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def t2i_action_prompt(
        self,
        image_ids,
        text_ids,
        masked_ids,
        action_dims,
        label_ids,
        device,
        state_ids,
        config=None,
    ):
        """Build prompt for image training.Important,all input have been offset"""
        max_text_len = self.max_text_len - 1
        B = len(text_ids)
        seqs, masks, labels = [], [], []
        for i in range(B):
            text = text_ids[i]
            action_dim = action_dims[i]
            if len(text) == 0:
                text = [self.text_tokenizer.bos_token_id]
            elif text[0] != self.text_tokenizer.bos_token_id:
                text = [self.text_tokenizer.bos_token_id] + text
            text = [int(self.sptids_dict["<|t2i|>"])] + text
            dim_token = "<|7dim|>" if action_dim == 7 else "<|14dim|>"
            if config and getattr(config.training, "t2i_ignore_state", False):
                state_block = []
            else:
                state_block = self.text_tokenizer("The current states of robot is:")[
                    "input_ids"
                ]
                state_block += [int(self.sptids_dict["<|sostate|>"])]
                state_block += state_ids[i].tolist()
                state_block += [int(self.sptids_dict["<|eostate|>"])]
            image_block = []
            images_description_text = [
                self.text_tokenizer("The third/head view of robot is:")["input_ids"],
            ]
            if action_dim == 7:
                images_description_text.append(
                    self.text_tokenizer("The wrist view of robot is:")["input_ids"]
                )
            else:
                images_description_text.append(
                    self.text_tokenizer("The left wrist view of robot is:")["input_ids"]
                )
                images_description_text.append(
                    self.text_tokenizer("The right wrist view of robot is:")[
                        "input_ids"
                    ]
                )
            for j, img in enumerate(
                image_ids[i]
            ):  # use action dim to match images num, need to be refine
                image_block += images_description_text[j]
                image_block += (
                    [int(self.sptids_dict["<|soi|>"])]
                    + img.tolist()
                    + [int(self.sptids_dict["<|eoi|>"])]
                )
            # if no prev_action_ids, will still add <|soa|>,<|eoa|> into the seq will add text description itf
            action_dim_block = self.text_tokenizer("Robot's action dim is:")[
                "input_ids"
            ] + [int(self.sptids_dict[dim_token])]

            seq = (
                text
                + state_block
                + image_block
                + action_dim_block
                + [self.text_tokenizer.eos_token_id]
            )
            total_seq_len = len(seq) + 2 + len(label_ids[i])
            if max_text_len >= total_seq_len:
                old_len = len(seq)
                seq = [self.pad_id] * (max_text_len - total_seq_len) + seq
                temp_masks = [0] * (max_text_len - total_seq_len) + [1] * total_seq_len
            else:
                # should add the eos token
                seq = seq[: max_text_len - len(label_ids[i]) - 3] + [
                    self.text_tokenizer.eos_token_id
                ]  # 3 = <|soi|><|eoi|><eos>
                temp_masks = [1] * max_text_len  # +2 for two special tokens
            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat(
                [
                    torch.tensor(seq).to(device),
                    self.sptids_dict["<|soi|>"].to(device),
                    label_ids[i].to(device),
                    self.sptids_dict["<|eoi|>"].to(device),
                ],
                dim=0,
            )

            temp_label_ids = torch.where(
                temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids
            )

            seq = torch.cat(
                [
                    torch.tensor(seq).to(device),
                    self.sptids_dict["<|soi|>"].to(device),
                    masked_ids[i].to(device),
                    self.sptids_dict["<|eoi|>"].to(device),
                ],
                dim=0,
            )

            temp_masks = torch.tensor(temp_masks).to(device)

            seqs.append(seq)
            masks.append(temp_masks)
            labels.append(temp_label_ids)
        # MAKE SURE that the len of each tensor is equal
        return (
            torch.stack(seqs, dim=0),
            torch.stack(masks, dim=0),
            torch.stack(labels, dim=0),
        )

    def mm2a_prompt(
        self,
        image_ids,
        text_ids,
        state_ids,
        prev_action_ids,
        action_ids,
        action_dims,
        action_label_ids,
        device,
        chunk_size=24,
        config=None,
    ):
        """Build prompt for action training.Important,all input have been offset"""
        B = len(text_ids)
        seqs, masks, labels = [], [], []
        for i in range(B):
            text = text_ids[i]
            action_dim = action_dims[i]
            if len(text) == 0:
                text = [self.text_tokenizer.bos_token_id]
            elif text[0] != self.text_tokenizer.bos_token_id:
                text = [self.text_tokenizer.bos_token_id] + text
            text = [int(self.sptids_dict["<|mm2a|>"])] + text
            dim_token = "<|7dim|>" if action_dim == 7 else "<|14dim|>"
            if config and getattr(config.training, "ignore_state", False):
                state_block = []
            else:
                state_block = self.text_tokenizer("The current states of robot is:")[
                    "input_ids"
                ]
                state_block += [int(self.sptids_dict["<|sostate|>"])]
                state_block += state_ids[i].tolist()
                state_block += [int(self.sptids_dict["<|eostate|>"])]

            image_block = []
            images_description_text = [
                self.text_tokenizer("The third/head view of robot is:")["input_ids"],
            ]
            if action_dim == 7:
                images_description_text.append(
                    self.text_tokenizer("The wrist view of robot is:")["input_ids"]
                )
            else:
                images_description_text.append(
                    self.text_tokenizer("The left wrist view of robot is:")["input_ids"]
                )
                images_description_text.append(
                    self.text_tokenizer("The right wrist view of robot is:")[
                        "input_ids"
                    ]
                )
            for j, img in enumerate(
                image_ids[i]
            ):  # use action dim to match images num, need to be refine
                image_block += images_description_text[j]
                image_block += (
                    [int(self.sptids_dict["<|soi|>"])]
                    + img.tolist()
                    + [int(self.sptids_dict["<|eoi|>"])]
                )

            action_dim_block = self.text_tokenizer("Robot's action dim is:")[
                "input_ids"
            ] + [int(self.sptids_dict[dim_token])]
            if prev_action_ids[i].numel() != 0:
                pre_action_block = self.text_tokenizer("Robot's previous actions are:")[
                    "input_ids"
                ]
                pre_action_block += [int(self.sptids_dict["<|soa|>"])]
                pre_action_block += prev_action_ids[i].tolist()
                pre_action_block += [int(self.sptids_dict["<|eoa|>"])]
            else:
                pre_action_block = []
            seq = (
                text
                + state_block
                + image_block
                + action_dim_block
                + pre_action_block
                + [self.text_tokenizer.eos_token_id]
            )
            if self.max_action_prompt_len >= len(seq):
                old_len = len(seq)
                seq = [self.pad_id] * (self.max_action_prompt_len - len(seq)) + seq
                temp_masks = [0] * (self.max_action_prompt_len - old_len) + [1] * (
                    old_len + action_ids[i].shape[-1] + 2
                )
            else:
                # should add the eos token
                seq = seq[: self.max_action_prompt_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
                temp_masks = [1] * (
                    len(seq) + action_ids[i].shape[-1] + 2
                )  # +2 for two special tokens
            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat(
                [
                    torch.tensor(seq).to(device),
                    self.sptids_dict["<|soa|>"].to(device),
                    action_label_ids[i].to(device),
                    self.sptids_dict["<|eoa|>"].to(device),
                ],
                dim=0,
            )

            temp_label_ids = torch.where(
                temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids
            )

            seq = torch.cat(
                [
                    torch.tensor(seq).to(device),
                    self.sptids_dict["<|soa|>"].to(device),
                    action_ids[i].to(device),
                    self.sptids_dict["<|eoa|>"].to(device),
                ],
                dim=0,
            )

            # sequence_ids: [pad]...[pad] <|mm2a|> <bos> text image state <eos><soa><eoa>
            temp_masks = torch.tensor(temp_masks).to(device)
            # if dim==7, we pad the rest of the block to match with 14
            if action_dim == 7:
                seq = torch.cat(
                    [
                        seq,
                        torch.full(
                            (chunk_size * 7,),
                            self.pad_id,
                            dtype=seq.dtype,
                            device=seq.device,
                        ),
                    ],
                    dim=0,
                )
                temp_masks = torch.cat(
                    [
                        temp_masks,
                        torch.zeros(
                            (chunk_size * 7,),
                            dtype=temp_masks.dtype,
                            device=temp_masks.device,
                        ),
                    ],
                    dim=0,
                )
                temp_label_ids = torch.cat(
                    [
                        temp_label_ids,
                        torch.full(
                            (chunk_size * 7,),
                            -100,
                            dtype=temp_label_ids.dtype,
                            device=temp_label_ids.device,
                        ),
                    ],
                    dim=0,
                )
            seqs.append(seq)
            masks.append(temp_masks)
            labels.append(temp_label_ids)
        # MAKE SURE that the len of each tensor is equal
        return (
            torch.stack(seqs, dim=0),
            torch.stack(masks, dim=0),
            torch.stack(labels, dim=0),
        )

    def mm2a_gen_prompt(
        self,
        image_ids,
        text_ids,
        state_ids,
        prev_action_ids,
        action_dims,
        device,
        chunk_size=24,
        mask_token_id=126336,
    ):
        """Build prompt for action generation.all input have been offset"""
        B = len(text_ids)
        seqs, masks, prompt_ids = [], [], []
        for i in range(B):
            text = text_ids[i]
            action_dim = action_dims[i]
            action_ids = torch.tensor(
                [mask_token_id] * (action_dim * chunk_size)
            )  # need to be refined
            if len(text) == 0:
                text = [self.text_tokenizer.bos_token_id]
            elif text[0] != self.text_tokenizer.bos_token_id:
                text = [self.text_tokenizer.bos_token_id] + text
            text = [int(self.sptids_dict["<|mm2a|>"])] + text

            dim_token = "<|7dim|>" if action_dim == 7 else "<|14dim|>"
            state_block = self.text_tokenizer("The current states of robot is:")[
                "input_ids"
            ]
            state_block += [int(self.sptids_dict["<|sostate|>"])]
            state_block += state_ids[i].tolist()
            state_block += [int(self.sptids_dict["<|eostate|>"])]
            image_block = []
            images_description_text = [
                self.text_tokenizer("The third/head view of robot is:")["input_ids"],
            ]
            if action_dim == 7:
                images_description_text.append(
                    self.text_tokenizer("The wrist view of robot is:")["input_ids"]
                )
            else:
                images_description_text.append(
                    self.text_tokenizer("The left wrist view of robot is:")["input_ids"]
                )
                images_description_text.append(
                    self.text_tokenizer("The right wrist view of robot is:")[
                        "input_ids"
                    ]
                )
            for j, img in enumerate(
                image_ids[i]
            ):  # use action dim to match images num, need to be refine
                image_block += images_description_text[j]
                image_block += (
                    [int(self.sptids_dict["<|soi|>"])]
                    + img.tolist()
                    + [int(self.sptids_dict["<|eoi|>"])]
                )

            action_dim_block = self.text_tokenizer("Robot's action dim is:")[
                "input_ids"
            ] + [int(self.sptids_dict[dim_token])]
            if prev_action_ids[i].numel() != 0:
                pre_action_block = self.text_tokenizer("Robot's previous actions are:")[
                    "input_ids"
                ]
                pre_action_block += [int(self.sptids_dict["<|soa|>"])]
                pre_action_block += prev_action_ids[i].tolist()
                pre_action_block += [int(self.sptids_dict["<|eoa|>"])]
            else:
                pre_action_block = []

            seq = (
                text
                + state_block
                + image_block
                + action_dim_block
                + pre_action_block
                + [self.text_tokenizer.eos_token_id]
            )
            if self.max_action_prompt_len >= len(seq):
                old_len = len(seq)
                seq = [self.pad_id] * (self.max_action_prompt_len - len(seq)) + seq
                temp_masks = [0] * (self.max_action_prompt_len - old_len) + [1] * (
                    old_len + action_ids.shape[-1] + 2
                )
            else:
                # should add the eos token
                seq = seq[: self.max_action_prompt_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
                temp_masks = [1] * (
                    len(seq) + action_ids.shape[-1] + 2
                )  # +2 for two special tokens
            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            prompt_id = len(seq)
            seq = torch.cat(
                [
                    torch.tensor(seq).to(device),
                    self.sptids_dict["<|soa|>"].to(device),
                    action_ids.to(device),
                    self.sptids_dict["<|eoa|>"].to(device),
                ],
                dim=0,
            )
            temp_masks = torch.tensor(temp_masks).to(device)
            # if dim==7, we pad the rest of the block to match with 14
            if action_dim == 7:
                seq = torch.cat(
                    [
                        seq,
                        torch.full(
                            (chunk_size * 7,),
                            self.pad_id,
                            dtype=seq.dtype,
                            device=seq.device,
                        ),
                    ],
                    dim=0,
                )
                temp_masks = torch.cat(
                    [
                        temp_masks,
                        torch.zeros(
                            (chunk_size * 7,),
                            dtype=temp_masks.dtype,
                            device=temp_masks.device,
                        ),
                    ],
                    dim=0,
                )
            seqs.append(seq)
            masks.append(temp_masks)
            prompt_ids.append(torch.tensor(prompt_id).to(device))
        # MAKE SURE that the len of each tensor is equal
        return (
            torch.stack(seqs, dim=0),
            torch.stack(masks, dim=0),
            torch.stack(prompt_ids, dim=0),
        )

    def t2i_gen_prompt(self, text_ids, image_ids):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = (
                [int(self.sptids_dict["<|t2i|>"])]
                + text_ids[i]
                + [self.text_tokenizer.eos_token_id]
            )
            if self.max_text_len >= len(temp_ids):
                old_len = len(temp_ids)
                temp_ids = [self.pad_id] * (
                    self.max_text_len - len(temp_ids)
                ) + temp_ids
                temp_masks = [0] * (self.max_text_len - old_len) + [1] * (
                    old_len + image_ids.shape[-1] + 2
                )
            else:
                # should add the eos token
                temp_ids = temp_ids[: self.max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
                temp_masks = [1] * (
                    len(temp_ids) + image_ids.shape[-1] + 2
                )  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat(
                [
                    torch.tensor(temp_ids).to(device),
                    self.sptids_dict["<|soi|>"].to(device),
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"].to(device),
                ],
                dim=0,
            )

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def t2i_action_gen_prompt(
        self,
        image_ids,
        text_ids,
        state_ids,
        prev_action_ids,
        action_dims,
        device,
        chunk_size=24,
        mask_token_id=126336,
        image_tokens_len=256,
        config=None,
    ):
        """Build prompt for image used in action. Important,all input have been offset"""
        max_text_len = self.max_text_len - 1
        B = len(text_ids)
        seqs, masks, labels = [], [], []
        for i in range(B):
            text = text_ids[i]
            action_dim = action_dims[i]
            image_masked_ids = torch.tensor([mask_token_id] * image_tokens_len)
            if len(text) == 0:
                text = [self.text_tokenizer.bos_token_id]
            elif text[0] != self.text_tokenizer.bos_token_id:
                text = [self.text_tokenizer.bos_token_id] + text
            text = [int(self.sptids_dict["<|t2i|>"])] + text
            dim_token = "<|7dim|>" if action_dim == 7 else "<|14dim|>"
            if config and getattr(config.training, "t2i_ignore_state", False):
                print("t2i_ignore_state:YES")
                state_block = []
            else:
                state_block = self.text_tokenizer("The current states of robot is:")[
                    "input_ids"
                ]
                state_block += [int(self.sptids_dict["<|sostate|>"])]
                state_block += state_ids[i].tolist()
                state_block += [int(self.sptids_dict["<|eostate|>"])]
            image_block = []
            images_description_text = [
                self.text_tokenizer("The third/head view of robot is:")["input_ids"],
            ]
            if action_dim == 7:
                images_description_text.append(
                    self.text_tokenizer("The wrist view of robot is:")["input_ids"]
                )
            else:
                images_description_text.append(
                    self.text_tokenizer("The left wrist view of robot is:")["input_ids"]
                )
                images_description_text.append(
                    self.text_tokenizer("The right wrist view of robot is:")[
                        "input_ids"
                    ]
                )
            for j, img in enumerate(
                image_ids[i]
            ):  # use action dim to match images num, need to be refine
                image_block += images_description_text[j]
                image_block += (
                    [int(self.sptids_dict["<|soi|>"])]
                    + img.tolist()
                    + [int(self.sptids_dict["<|eoi|>"])]
                )
            # if no prev_action_ids, will still add <|soa|>,<|eoa|> into the seq will add text description itf
            action_dim_block = self.text_tokenizer("Robot's action dim is:")[
                "input_ids"
            ] + [int(self.sptids_dict[dim_token])]

            seq = (
                text
                + state_block
                + image_block
                + action_dim_block
                + [self.text_tokenizer.eos_token_id]
            )
            total_seq_len = len(seq) + 2 + len(image_masked_ids)
            if max_text_len >= total_seq_len:
                seq = [self.pad_id] * (max_text_len - total_seq_len) + seq
                temp_masks = [0] * (max_text_len - total_seq_len) + [1] * total_seq_len
            else:
                # should add the eos token
                seq = seq[: max_text_len - len(image_masked_ids) - 3] + [
                    self.text_tokenizer.eos_token_id
                ]  # 3 = <|soi|><|eoi|><eos>
                temp_masks = [1] * max_text_len  # +2 for two special tokens
            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]

            seq = torch.cat(
                [
                    torch.tensor(seq).to(device),
                    self.sptids_dict["<|soi|>"].to(device),
                    image_masked_ids.to(device),
                    self.sptids_dict["<|eoi|>"].to(device),
                ],
                dim=0,
            )

            temp_masks = torch.tensor(temp_masks).to(device)

            seqs.append(seq)
            masks.append(temp_masks)
        # MAKE SURE that the len of each tensor is equal
        return (torch.stack(seqs, dim=0), torch.stack(masks, dim=0))

    # language modeling
    def lm_prompt(self, text_ids, max_seq_len):
        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                temp_labels_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (
                    max_seq_len - len(temp_ids)
                )
                temp_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (
                    max_seq_len - len(temp_ids)
                )
                temp_masks = [1] * len(temp_ids) + [0] * (max_seq_len - len(temp_ids))
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_masks = torch.tensor(temp_masks)
            temp_labels_ids = torch.tensor(temp_labels_ids)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(attention_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    # language modeling
    def lm_chat_prompt(self, text_ids, max_seq_len):
        sequence_ids = []
        prompt_masks = []
        label_ids = []

        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                temp_labels_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (
                    max_seq_len - len(temp_ids)
                )
                temp_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (
                    max_seq_len - len(temp_ids)
                )
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]

            end_header_id = int(self.sptids_dict["<|end_header_id|>"])
            end_header_pos = -1
            for pos in range(
                len(temp_ids) - 1, -1, -1
            ):  # 尝试从文本序列中寻找<|end_header_id|>
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = end_header_pos + 1
            else:
                prompt_length = 0
            temp_masks = [1] * prompt_length + [0] * (len(temp_ids) - prompt_length)

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_masks = torch.tensor(temp_masks)
            temp_labels_ids = torch.tensor(temp_labels_ids)
            sequence_ids.append(temp_ids.unsqueeze(0))
            prompt_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(prompt_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def mmu_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        prompt_masks = []
        label_ids = []
        max_text_len = self.max_text_len - 1
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (
                    max_text_len - len(temp_ids)
                )
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3) + [0] * (
                    max_text_len - len(temp_ids)
                )
            else:
                # should add the eos token
                temp_ids = temp_ids[: max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
                temp_masks = [1] * (
                    len(temp_ids) + image_ids.shape[-1] + 3
                )  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat(
                [
                    torch.tensor([self.ignore_id]).to(device),
                    torch.tensor([self.ignore_id]).to(device),
                    torch.ones_like(image_ids[i]) * self.ignore_id,
                    torch.tensor([self.ignore_id]).to(device),
                    torch.tensor(temp_ids).to(device),
                ],
                dim=0,
            )

            temp_label_ids = torch.where(
                temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids
            )

            return_temp_ids = torch.cat(
                [
                    self.sptids_dict["<|mmu|>"].to(device),  # task token
                    self.sptids_dict["<|soi|>"].to(device),
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"].to(device),
                    torch.tensor(temp_ids).to(device),
                ],
                dim=0,
            )
            end_header_id = int(self.sptids_dict["<|end_header_id|>"])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = (
                    len(return_temp_ids) - len(temp_ids) + end_header_pos + 1
                )
            else:
                prompt_length = len(return_temp_ids) - len(temp_ids)
            predict_length = len(return_temp_ids) - prompt_length
            prompt_mask = [1] * prompt_length + [0] * predict_length
            prompt_mask = torch.tensor(prompt_mask).to(device)
            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(prompt_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def mmu_gen_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        prompt_masks = []
        max_text_len = self.max_text_len - 1
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (
                    max_text_len - len(temp_ids)
                )
            else:
                # should add the eos token
                temp_ids = temp_ids[: max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]

            # print(f"mmu temp_ids: {temp_ids}")
            return_temp_ids = torch.cat(
                [
                    self.sptids_dict["<|mmu|>"].to(device),  # task token
                    self.sptids_dict["<|soi|>"].to(device),
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"].to(device),
                    torch.tensor(temp_ids).to(device),
                ],
                dim=0,
            )

            end_header_id = int(self.sptids_dict["<|end_header_id|>"])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = (
                    len(return_temp_ids) - len(temp_ids) + end_header_pos + 1
                )
            else:
                prompt_length = len(return_temp_ids) - len(temp_ids)
            predict_length = len(temp_ids) - prompt_length
            print(
                f"prompt_length: {prompt_length}, predict_length: {predict_length}, all length: {len(return_temp_ids)}, {return_temp_ids[-predict_length:]}"
            )
            prompt_mask = [1] * prompt_length + [0] * predict_length
            prompt_mask = torch.tensor(prompt_mask).to(device)
            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
        return torch.cat(sequence_ids, dim=0), torch.cat(prompt_masks, dim=0)

    def mmu_action_prompt(
        self,
        image_ids,
        task_ids,
        description_ids,
        action_dims,
        device,
        state_ids,
        mmu_learn_eos: bool = False,
        config=None,
    ):
        """
        Build prompt for text used in action.
        """
        sequence_ids, prompt_masks, label_ids = [], [], []
        max_text_len = self.max_text_len - 1

        for i in range(len(task_ids)):

            if config and getattr(config.training, "mmu_ignore_state", False):
                state_block = []
            else:
                # print("Include state")
                state_block = self.text_tokenizer("The current states of robot is:")[
                    "input_ids"
                ]
                state_block += [int(self.sptids_dict["<|sostate|>"])]
                state_block += state_ids[i].tolist()
                state_block += [int(self.sptids_dict["<|eostate|>"])]

            # ===== image description blocks =====
            image_block = []
            action_dim = action_dims[i]
            images_description_text = [
                self.text_tokenizer("The third/head view of robot is:")["input_ids"],
            ]
            if action_dim == 7:
                images_description_text.append(
                    self.text_tokenizer("The wrist view of robot is:")["input_ids"]
                )
            else:
                images_description_text.append(
                    self.text_tokenizer("The left wrist view of robot is:")["input_ids"]
                )
                images_description_text.append(
                    self.text_tokenizer("The right wrist view of robot is:")[
                        "input_ids"
                    ]
                )
            for j, img in enumerate(image_ids[i]):
                image_block += images_description_text[j]
                image_block += (
                    [int(self.sptids_dict["<|soi|>"])]
                    + img.tolist()
                    + [int(self.sptids_dict["<|eoi|>"])]
                )

            # ===== whole text =====
            task_part = task_ids[i]
            if len(task_part) == 0:
                task_part = [self.text_tokenizer.bos_token_id]
            elif task_part[0] != self.text_tokenizer.bos_token_id:
                task_part = [self.text_tokenizer.bos_token_id] + task_part
            task_part = [int(self.sptids_dict["<|mmu|>"])] + task_part

            desc_part = description_ids[i] + [self.text_tokenizer.eos_token_id]

            # ===== action dim token =====
            dim_token = "<|7dim|>" if action_dims[i] == 7 else "<|14dim|>"
            action_dim_block = self.text_tokenizer("Robot's action dim is:")[
                "input_ids"
            ] + [int(self.sptids_dict[dim_token])]

            text_tokens = (
                task_part + state_block + image_block + action_dim_block + desc_part
            )
            ignore_len = len(text_tokens) - len(desc_part)
            # ===== clip or pad =====
            if max_text_len >= len(text_tokens):
                text_tokens = text_tokens + [self.text_tokenizer.eos_token_id] * (
                    max_text_len - len(text_tokens)
                )
            else:
                text_tokens = text_tokens[: max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]
            # ===== constuct label_ids =====
            if config and getattr(config.training, "mmu_learn_eos", False):
                temp_label_ids = [self.ignore_id] * ignore_len + text_tokens[
                    ignore_len:
                ]
            else:
                temp_label_ids = (
                    [self.ignore_id] * ignore_len
                    + desc_part
                    + [self.ignore_id]
                    * (len(text_tokens) - ignore_len - len(desc_part))
                )
            temp_label_ids = torch.tensor(temp_label_ids).to(device)

            # ===== constuct return_temp_ids =====
            return_temp_ids = torch.tensor(text_tokens).to(device)

            # ===== prompt mask: prompt=1,desc=0 =====
            prompt_length = ignore_len
            predict_length = len(text_tokens) - ignore_len
            prompt_mask = torch.tensor([1] * prompt_length + [0] * predict_length).to(
                device
            )
            # ===== batch =====
            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(prompt_masks, dim=0),
            torch.cat(label_ids, dim=0),
        )

    def r2i_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        prompt_masks = []
        label_ids = []
        r2i_id = int(self.sptids_dict["<|r2i|>"])
        soi_id = int(self.sptids_dict["<|soi|>"])
        eoi_id = int(self.sptids_dict["<|eoi|>"])
        max_text_len = self.max_text_len - 1  # 512，include BOS text EOS
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            text_ids_with_bos_eos = text_ids[i] + [self.text_tokenizer.eos_token_id]
            if max_text_len >= len(text_ids_with_bos_eos):
                # minus 1 because task token was prepended to the former image tokens
                text_ids_full_len = text_ids_with_bos_eos + [
                    self.text_tokenizer.eos_token_id
                ] * (max_text_len - len(text_ids_with_bos_eos))
            else:
                # should add the eos token
                text_ids_full_len = text_ids_with_bos_eos[: max_text_len - 1] + [
                    self.text_tokenizer.eos_token_id
                ]

            sequence_ids.append(
                torch.cat(
                    [
                        torch.tensor([r2i_id]).to(device),  # task token
                        torch.tensor(text_ids_full_len).to(device),
                        torch.tensor([soi_id]).to(device),
                        image_ids[i],
                        torch.tensor([eoi_id]).to(device),
                    ],
                    dim=0,
                ).unsqueeze(0)
            )

            end_header_id = int(self.sptids_dict["<|end_header_id|>"])
            end_header_pos = -1
            for pos in range(len(text_ids_full_len) - 1, -1, -1):
                if text_ids_full_len[pos] == end_header_id:
                    end_header_pos = pos
                    break
            prompt_mask = torch.zeros(sequence_ids[i].size(1)).to(device)
            prompt_mask[0] = 1  # task_id
            if end_header_pos != -1:
                prompt_mask[1 : end_header_pos + 2] = 1
            else:
                prompt_mask[1 : len(text_ids_full_len) + 1] = 1
            prompt_mask[len(text_ids_full_len) + 1] = 1
            prompt_mask[len(text_ids_full_len) + 2 + len(image_ids[i])] = 1
            prompt_masks.append(prompt_mask.unsqueeze(0))

        return (
            torch.cat(sequence_ids, dim=0),
            torch.cat(prompt_masks, dim=0),
            torch.cat(sequence_ids, dim=0),
        )

    def mask_prompt(self):
        pass

    def __call__(self, input, task, padding=True, config=None):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "t2i":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids, input[2])

        # elif task == "t2v":
        #     text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
        #     image_ids = input[1]  # (B, #tokens)
        #     sequence_ids_with_masks = self.t2v_prompt(text_ids, image_ids, input[2])

        elif task == "t2i_plus_lm":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(
                text_ids[: config.training.batch_size], image_ids, input[2]
            )
            sequence_ids_with_masks_lm = self.lm_prompt(
                text_ids[config.training.batch_size :], input[3]
            )
            return sequence_ids_with_masks, sequence_ids_with_masks_lm

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids, image_ids)
        elif task == "mm2a_gen":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])["input_ids"]
            state_ids = input[2]
            prev_action_ids = input[3]  # None if None
            action_dim = input[4]
            device = input[5]
            chunk_size = input[6]
            sequence_ids_with_masks = self.mm2a_gen_prompt(
                image_ids,
                text_ids,
                state_ids,
                prev_action_ids,
                action_dim,
                device,
                chunk_size=chunk_size,
            )
        elif task == "mm2a":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])["input_ids"]
            state_ids = input[2]
            prev_action_ids = input[3]
            action_ids = input[4]
            action_dim = input[5]
            label_list = input[6]
            device = input[7]
            chunk_size = input[8]
            sequence_ids_with_masks = self.mm2a_prompt(
                image_ids,
                text_ids,
                state_ids,
                prev_action_ids,
                action_ids,
                action_dim,
                label_list,
                device,
                chunk_size=chunk_size,
                config=config,
            )
        elif task == "mmu_action":
            image_ids = input[0]
            task_ids = self.text_tokenizer(input[1])["input_ids"]
            description_ids = self.text_tokenizer(input[2])["input_ids"]
            action_dim = input[3]
            device = input[4]
            state_tokens = input[5]
            sequence_ids_with_masks = self.mmu_action_prompt(
                image_ids,
                task_ids,
                description_ids,
                action_dim,
                device,
                state_tokens,
                config=config,
            )
        elif task == "t2i_action":
            image_ids = input[0]
            task_ids = self.text_tokenizer(input[1])["input_ids"]
            masked_ids = input[2]
            action_dim = input[3]
            label_ids = input[4]
            device = input[5]
            state_tokens = input[6]
            sequence_ids_with_masks = self.t2i_action_prompt(
                image_ids,
                task_ids,
                masked_ids,
                action_dim,
                label_ids,
                device,
                state_tokens,
                config=config,
            )
        elif task == "t2i_action_gen":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])["input_ids"]
            state_ids = input[2]
            prev_action_ids = input[3]  # None if None
            action_dim = input[4]
            device = input[5]
            chunk_size = input[6]
            sequence_ids_with_masks = self.t2i_action_gen_prompt(
                image_ids,
                text_ids,
                state_ids,
                prev_action_ids,
                action_dim,
                device,
                chunk_size=chunk_size,
                config=config,
            )
        # elif task == "t2v_gen":
        #     text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
        #     image_ids = input[1]  # (B, #tokens)
        #     sequence_ids_with_masks = self.t2v_gen_prompt(text_ids, image_ids)
        elif task == "lm":
            text_ids = self.text_tokenizer(input[0], truncation=True)[
                "input_ids"
            ]  # (B, max_len)
            sequence_ids_with_masks = self.lm_prompt(text_ids, input[1])

        elif task == "lm_chat":
            text_ids = self.text_tokenizer(input[0], truncation=True)[
                "input_ids"
            ]  # (B, max_len)
            sequence_ids_with_masks = self.lm_chat_prompt(text_ids, input[1])

        elif task == "mmu":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])["input_ids"]
            sequence_ids_with_masks = self.mmu_prompt(image_ids, text_ids)

        elif task == "r2i":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])["input_ids"]
            sequence_ids_with_masks = self.r2i_prompt(image_ids, text_ids)

        else:
            raise NotImplementedError

        return sequence_ids_with_masks


if __name__ == "__main__":
    pass
