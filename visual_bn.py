
torch.save(base_model.module.MAE_encoder.encoder.first_conv[1].running_mean.cpu(), "first_conv_running_mean_updated.pth")
torch.save(base_model.module.MAE_encoder.encoder.first_conv[1].running_var.cpu(), "first_conv_running_var_updated.pth")
torch.save(base_model.module.MAE_encoder.encoder.second_conv[1].running_mean.cpu(), "second_conv_running_mean_updated.pth")
torch.save(base_model.module.MAE_encoder.encoder.second_conv[1].running_var.cpu(), "second_conv_running_var_updated.pth")



torch.save(base_model.module.class_head[1].running_mean.cpu(), "class_head_first_running_mean_updated.pth")
torch.save(base_model.module.class_head[1].running_var.cpu(), "class_head_first_running_var_updated.pth")
torch.save(base_model.module.class_head[5].running_mean.cpu(), "class_head_second_running_mean_updated.pth")
torch.save(base_model.module.class_head[5].running_var.cpu(), "class_head_second_running_var_updated.pth")