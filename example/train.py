
import sucrose



sucrose.start_project('example', 'test1')


writer = sucrose.start_pytorch_tensorboard()


for epoch in range(100):

    # train
    ...
    writer.add_scalar()
    sucrose.save_ckpts(epoch, {'model': 100})

    # eval

