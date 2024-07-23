from safetensors.torch import save_file, load_file

def save_model_weights(model, path):
    state_dict = model.state_dict()
    save_file(state_dict, path)

def load_model_weights(model, path):
    state_dict = load_file(path)
    model.load_state_dict(state_dict)