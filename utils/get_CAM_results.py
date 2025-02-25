import torch


def choose_layer(model_name, model):
    # choose the target layer for CAM
    if model_name == 'DCNN':
        target_layer = model.layer4[3]
    elif (model_name == 'ResNet'
          or model_name == 'WKN_Laplace'
          or model_name == 'Inception'
          or model_name == 'DRSN'):
        target_layer = model.Iden
    elif model_name == 'AlexNet':
        target_layer = model.features[12]
    else:
        raise Exception("Please define the target layer for this model: {}"
                        .format(model_name))
    return target_layer


class BaseCAM:
    def __init__(self):
        pass

    def __call__(self, CAM_type, activations, gradients):
        # get the weight of each feature map
        weights = self.get_cam_weights(activations,
                                       gradients,
                                       CAM_type)[:, :, None]
        cam = weights * activations
        cam = torch.clamp(cam, min=0)
        cam = torch.sum(cam, dim=1)  # operation in the channel dimension
        min = torch.min(cam, dim=1).values.unsqueeze(1)
        max = torch.max(cam, dim=1).values.unsqueeze(1)
        Lc = (cam - min) / (max - min + 1e-20)
        return Lc


class GradCAM(BaseCAM):
    def __init__(self):
        super().__init__()

    def get_cam_weights(self,
                        activations,
                        grads,
                        CAM_type):

        if CAM_type == 'GradCAM':
            weights = torch.sum(grads, dim=2)

        elif CAM_type == 'GradCAMPP':
            grads_power_2 = grads.pow(2)
            grads_power_3 = grads.pow(3)
            sum_activations = torch.sum(activations, dim=2)
            eps = 1e-20
            ai = grads_power_2 / (2 * grads_power_2
                                  + sum_activations[:, :, None]
                                  * grads_power_3
                                  + eps)
            ai = torch.where(grads != 0, ai, 0)
            relu_grads = torch.clamp(grads, min=0)
            weights = relu_grads * ai
            weights = torch.sum(weights, dim=2)

        elif CAM_type == 'PFM':  # pure feature maps
            weights = torch.ones(
                activations.shape[0],
                activations.shape[1]
                ).cuda()

        return weights
