## Função auxiliar para dividir o array de dados vindos do banco
# em treino e teste, o limit_of_data é mais pra desenvolvimento
# para poder pegar os 100 primeiros de forma mais tranquila


def split_data(data, train_percentage, limit_of_data=None):
    if not isinstance(train_percentage, int) or isinstance(train_percentage, bool):
        raise TypeError("train_percentage should be an int like 95")

    if train_percentage < 1 or train_percentage > 99:
        raise ValueError("train_percentage should be an int between 1 and 99")

    if limit_of_data is not None:
        if not isinstance(limit_of_data, int) or isinstance(limit_of_data, bool):
            raise TypeError("limit_of_data should be an int like 1000")

        if limit_of_data < 1:
            raise ValueError("limit_of_data should be at least 1")

        limit_of_data = min(limit_of_data, len(data))
        used_data = data[:limit_of_data]
    else:
        used_data = data

    train_data_off_set = (len(used_data) * train_percentage) // 100

    # anula a possibilidade de treino ou teste virem vazios
    if not (1 < train_data_off_set < len(used_data)):
        raise ValueError("train_percentage and limit_of_data was too short")

    train_data = used_data[:train_data_off_set]
    test_data = used_data[train_data_off_set:]

    return (train_data, test_data)
