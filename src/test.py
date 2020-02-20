def test(config):
    print('-Test:loading dataset...')
    DuConv_DataSet=My_dataset(config.run_type,config.data_dir,config.voc_embedding_save_path)
    train_loader = DataLoader(dataset=DuConv_DataSet,\
            shuffle=True, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
    print('-Test:building models...')
    checkpoint =torch.load(config.continue_training)  if config.continue_training != " " else None
    encoder,decoder=build_models(DuConv_DataSet.voc,config,checkpoint)
    print('-Initializing test process...')
