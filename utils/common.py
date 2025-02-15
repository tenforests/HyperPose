from PIL import Image
import matplotlib.pyplot as plt


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
    display_count = len(log_hooks)
    n_outputs =  1
    fig = plt.figure(figsize=(16 + (n_outputs * 4), 6 * display_count))
    gs = fig.add_gridspec(display_count, (3 + n_outputs))
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        vis_faces_iterative(hooks_dict, fig, gs, i)
    plt.tight_layout()
    return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
                                                     float(hooks_dict['diff_target'])))
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))



def vis_faces_iterative(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['ref_img'])
    plt.title('Input ref\n')
    fig.add_subplot(gs[i, 1])
    # plt.imshow(hooks_dict['w_inversion'])
    # plt.title('W-Inversion\n')
    plt.imshow(hooks_dict['pose_img'])
    plt.title('pose_img\n')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['target_img'])
    plt.title('Target\nOut={:.2f}'.format( float(hooks_dict['diff_target'])))
    fig.add_subplot(gs[i, 3])
    plt.imshow(hooks_dict['output_img'])
    plt.title('output')
    # for idx, output_idx in enumerate(range(len(hooks_dict['output_face']) - 1, -1, -1)):
    #     output_image, similarity = hooks_dict['output_face'][output_idx]
    #     fig.add_subplot(gs[i, 3 + idx])
    #     plt.imshow(output_image)
    #     plt.title('Output {}\n Target Sim={:.2f}'.format(output_idx, float(similarity)))
