import pickle
from language_modeling.domain.modeling.utils.decoder.greedy_decoder import GreedyDecoder


def main():
    with open("preprocessor_and_model.bin", "rb") as file:
        prep = pickle.load(file)

    decoder = GreedyDecoder(prep[0]._tokenizer, prep[1])
    output = decoder.generate_text("Les taux", 12, 2)
    print(output)


if __name__ == "__main__":
    main()
