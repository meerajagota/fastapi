

from huggingface_hub import login
login(token="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")

class MistralInstruct():
    def __init__(
        self,
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_path="mistralai/Mistral-7B-Instruct-v0.1-split",
        batch_size=1,
        tp_degree=2,
        n_positions=8000,
        amp="bf16",
        **model_kwargs
    ):
        super().__init__()

        self.model_id = model_id
        self.model_path = model_path

        self.batch_size = batch_size
        self.tp_degree = tp_degree
        self.n_positions = n_positions
        self.amp = amp
        self.model_kwargs = model_kwargs

        self.model_cpu = None
        self.model_neuron = None
        self.tokenizer = None

    def _load_pretrained_model(self):
        from transformers import AutoModelForCausalLM
        from transformers_neuronx.module import save_pretrained_split

        self.model_cpu = AutoModelForCausalLM.from_pretrained(self.model_id)
        save_pretrained_split(self.model_cpu, self.model_path)

    def _load_neuron_model(self):
        from transformers_neuronx import constants
        from transformers_neuronx.config import NeuronConfig
        from transformers_neuronx.mistral.model import MistralForSampling
        

        if self.model_cpu is None:
            # Load and save the CPU model
            self._load_pretrained_model()


        # Create and compile the Neuron model
        self.model_neuron = MistralForSampling.from_pretrained(
            self.model_path,
            batch_size=self.batch_size,
            tp_degree=self.tp_degree,
            n_positions=self.n_positions,
            amp=self.amp,
            context_length_estimate=8000,
            **self.model_kwargs
        )
        self.model_neuron.to_neuron()

        self.model_neuron.save("model.pt")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)


    def generate(self, messages: list, return_tensors="pt", sequence_length=256):
        import torch

        if self.tokenizer is None:
            self._load_tokenizer()

        if self.model_neuron is None:
            self._load_neuron_model()

        encodeds = self.tokenizer.apply_chat_template(
            messages, return_tensors=return_tensors
        )

        # Run inference
        with torch.inference_mode():
            generated_sequence = self.model_neuron.sample(
                encodeds,sequence_length=sequence_length, start_ids=None
            )

        return [self.tokenizer.decode(tok) for tok in generated_sequence]


if __name__ == "__main__":
    

    remote_instruct_model = MistralInstruct()
    

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {
            "role": "assistant",
            "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount "
            "of zesty flavour to whatever I'm cooking up in the kitchen!",
        },
        {"role": "user", "content": "Do you have mayonnaise recipes?"},
    ]


    chat_completion = remote_instruct_model.generate(messages)
    print(chat_completion)

    
    
