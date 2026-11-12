from rfdetr import RFDETRMedium

if __name__ == '__main__':
    model = RFDETRMedium()

    model.train(
        dataset_dir="data/det_dataset",
        epochs=40,
        batch_size=4,              # если OOM: 2
        output_dir="runs/finetune/rfdetr_medium_ft_aug",
    )
    print("Чекпоинты: runs/finetune/rfdetr_medium_ft_aug")