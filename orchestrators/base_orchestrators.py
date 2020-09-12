import time
import torch
from models.model_builder import build_model
from utils import logger_utils


def start_training(EPOCHS, device, train_loader, test_loader, **models_dict):
    results = {}
    logger.info("\n**** Started training ****\n")
    for model_type in models_dict:
        #print(f"Model: {model_type}")
        logger.info(f"\nModel: {model_type}\n")
        train_accs, train_losses, test_acc, test_losses, best_model = build_model(EPOCHS, device, train_loader, test_loader, **models_dict[model_type])
        results[model_type] = [train_accs, train_losses, test_acc, test_losses, best_model]
        #print(results)
        logger.info(f"\nresults : {results}\n")
        time.sleep(10)
    logger.info("\n**** Ended training ****\n")
    return results

def display_classwise_accuracy(test_loader,device,model):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels.to(device)).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        logger.info(f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]}')