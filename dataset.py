# Imports

from torch.utils.data import Dataset
from datasets import load_dataset

from transformers import AutoTokenizer

# Set path to dataset and path to the SFT model folder

path = "/content/drive/MyDrive/NLP/Method1_dataset.csv"
tokenizer_path = '/content/drive/MyDrive/NLP/SFT_GPT-2M_Dolly15k'


# Inherits from torch.utils.data
class SubredditQuestionDataset(Dataset):
    '''This class loads the dataset, tokenizes it according to the selected model,
    and block size parameters. Initializes the object with a torch.utils.data.Dataset instance.'''

    def __init__(self, path, block_size=512, num_records=512, tokenizer_path='/content/drive/MyDrive/SFT_GPT-2M_Dolly15k'):
        super().__init__()
        dataset = load_dataset('csv', data_files=path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding=True, max_length=block_size, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

        print('Loading the dataset')

        def tokenize_this(sample):
            sample['input_ids'] = tokenizer.encode(sample['Question'])
            return sample

        dataset = dataset.remove_columns(['Subreddit'])
        dataset = dataset['train'].select(range(num_records))
        dataset = dataset.map(tokenize_this, batched=False)
        dataset = dataset.rename_columns({'Question': 'query'})
        dataset.set_format(type="torch")

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def get_dataset(path, tokenizer_path, block_size=512, num_records=512):
    subreddit_question_dataset = SubredditQuestionDataset(
        path, block_size=block_size, num_records=num_records, tokenizer_path=tokenizer_path)

    return subreddit_question_dataset
