from VITTrain import VisionTransformerTrainer

# Hyperparameters
learning_rate = 1e-5
batch_size = 32
num_epochs = 1
weight_decay = 0.01
momentum = 0.9
dropout_rate = 0.3

trainer = VisionTransformerTrainer(
    train_path='./train',
    val_path='./valid',
    test_path='./test',
    plot_output=True,
    batch_size = batch_size,
    num_epochs = num_epochs,
    learning_rate = learning_rate
)
trainer.train()
trainer.evaluate_test_set()
