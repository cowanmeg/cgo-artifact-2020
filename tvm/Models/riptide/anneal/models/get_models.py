from .resnet import resnet18

def get_model(name, **kwargs):
    models = {
         'resnet18': resnet18,
    }

    if 'resnet' in name:
        name = 'resnet18'
    elif name not in models:
        raise ValueError("%s Not in supported models.\n\t%s" %
                         (name, '\n\t'.join(sorted(models.keys()))))
    model = models[name](**kwargs)
    return model
