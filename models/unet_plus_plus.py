import segmentation_models_pytorch as smp

def unet_plus_plus(encoder_name, encoder_weights, in_channels, classes, activation=None):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )

    return model