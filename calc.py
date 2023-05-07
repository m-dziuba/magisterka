from main import training_loop
for model in ["basic", "in_series", "cnn_lstm"]:
    training_loop(model)

