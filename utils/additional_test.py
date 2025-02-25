import torch
from dataset_processing import dataset_processing_module


def get_accuracy_for_noised(datasets, phase, args, snr, model):
    if args.model_name == 'AlexNet':
        model.eval()  # AlexNet has Dropout layers
    else:
        model.train()  # Other models have BN layers
    num_test_samples = len(datasets[phase].labels)
    dataloaders = torch.utils.data.DataLoader(
        datasets[phase],
        batch_size=num_test_samples,
        shuffle=False,
        pin_memory=True
        )
    for _, (inputs, labels) in enumerate(dataloaders):
        # adding noise
        for i in range(num_test_samples):
            inputs[i, :] = dataset_processing_module.add_noise_perSNR(
                inputs[i, :].cpu(),
                snr
                )
        inputs = torch.unsqueeze(inputs, 1).cuda()
        labels = labels.cuda()
    accuracy = get_pred_accuracy(model, inputs, labels, num_test_samples)
    return accuracy


def get_pred_accuracy(model, inputs, labels, num_test_samples):
    logits = model(inputs)
    pred = logits.argmax(dim=1)
    correct = torch.eq(pred, labels).float().sum().item()
    accuracy = correct / num_test_samples
    return accuracy
