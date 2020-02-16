import pickle
from torchtext import data


class DGA2019(data.Dataset):
    def __init__(self, data_path: str, text_field: data.Field, label_field: data.LabelField, test_mode: bool = True):
        # Define tokenize for Urls

        self._text_field = text_field
        self._label_field = label_field

        # Length of DataSet
        self._ds_len = 0

        # Initialize Fields
        fields = [("url", self._text_field),
                  ("label", self._label_field)]
        examples = []

        print('read data_process from:{}'.format(data_path))
        with open(data_path, "rb") as f:
            urls_data, label_data = pickle.load(f)
        self._ds_len = len(urls_data)

        # Initialize Examples
        if test_mode:
            for url in urls_data:
                examples.append(data.Example.fromlist([url, None], fields))
        else:
            for url, label in zip(urls_data, label_data):
                examples.append(data.Example.fromlist([url, label], fields))

        # 调用super调用父类构造方法，产生标准Dataset
        super(DGA2019, self).__init__(examples, fields)

    def __len__(self):
        return self._ds_len
