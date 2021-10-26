import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_sequence(model, tokenizer, seed: str, max_length: int):
    model.eval()
    model.to(device)

    SoftMax = torch.nn.Softmax(0)

    input = torch.tensor(
        tokenizer.encode("<SOS> " + seed).ids,
        dtype=torch.long,
        device=device
    )
    input = input.unsqueeze(0)
    output = input.cpu().tolist()
    
    for _ in range(max_length):
        with torch.no_grad():
            hiddens = model.init_hiddens(1)

            logits, hiddens = model(
                torch.tensor(input, dtype=torch.long, device=device),
                hiddens
            )
            _, top = torch.topk(
                SoftMax(
                    logits[0][-1]
                ),
                1
            )

            input = top.unsqueeze(0)

            output.append(top.item())
            
    return output