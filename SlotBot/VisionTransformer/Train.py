from VIT_Trainer import VisionTransformerTrainer

# Hyperparameters
learning_rate = 1e-5
batch_size = 32
num_epochs = 15
weight_decay = 0.01

trainer = VisionTransformerTrainer(
    train_path='./ver6/train',
    val_path='./ver6/valid',
    test_path='./ver6/test',
    plot_output=True,
    batch_size = batch_size,
    num_epochs = num_epochs,
    learning_rate = learning_rate,
    save_dir = './VITrun_ver6'
)
trainer.train()
trainer.evaluate_test_set()
