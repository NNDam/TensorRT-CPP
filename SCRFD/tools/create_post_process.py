import torch
import torch.nn as nn

input_height = 640
input_width  = 640

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box (Batch processing).

    Args:
        points (Tensor): Shape (b, n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, :, 0] - distance[:, :, 0]
    y1 = points[:, :, 1] - distance[:, :, 1]
    x2 = points[:, :, 0] + distance[:, :, 2]
    y2 = points[:, :, 1] + distance[:, :, 3]
    if max_shape is not None:
        x1 = torch.clamp(x1, min=0, max=max_shape[1])
        y1 = torch.clamp(y1, min=0, max=max_shape[0])
        x2 = torch.clamp(x2, min=0, max=max_shape[1])
        y2 = torch.clamp(y2, min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], dim=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (b, n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    batch_size, n_boxes, _ = distance.shape
    tiled_points = points.repeat(1,1,5)#.reshape((batch_size, n_boxes, -1, 2))
    return tiled_points + distance

class SCRFDPostProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        
        width_8  = torch.arange(input_height // 8)
        height_8 = torch.arange(input_width // 8)
        width_16  = torch.arange(input_height // 16)
        height_16 = torch.arange(input_width // 16)
        width_32  = torch.arange(input_height // 32)
        height_32 = torch.arange(input_width // 32)

        # Calculate Anchor
        self._num_anchors = 2

        anchor_centers_8 = torch.stack(torch.meshgrid([width_8, height_8])[::-1], dim=-1)
        anchor_centers_8 = (anchor_centers_8 * 8).reshape((-1, 2))
        self.anchor_centers_8 = torch.stack([anchor_centers_8]*self._num_anchors, dim=1).reshape((-1,2))

        anchor_centers_16 = torch.stack(torch.meshgrid([width_16, height_16])[::-1], dim=-1)
        anchor_centers_16 = (anchor_centers_16 * 16).reshape((-1, 2))
        self.anchor_centers_16 = torch.stack([anchor_centers_16]*self._num_anchors, dim=1).reshape((-1,2))

        anchor_centers_32 = torch.stack(torch.meshgrid([width_32, height_32])[::-1], dim=-1)
        anchor_centers_32 = (anchor_centers_32 * 32).reshape((-1, 2))
        self.anchor_centers_32 = torch.stack([anchor_centers_32]*self._num_anchors, dim=1).reshape((-1,2))


    def forward(self, score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32):
        # Tile anchor by batchsize
        batch_size = score_8.shape[0]
        tiled_anchor_centers_8  = self.anchor_centers_8.repeat(batch_size, 1, 1)   # Nx4 -> BxNx4
        tiled_anchor_centers_16 = self.anchor_centers_16.repeat(batch_size, 1, 1)   # Nx4 -> BxNx4
        tiled_anchor_centers_32 = self.anchor_centers_32.repeat(batch_size, 1, 1)   # Nx4 -> BxNx4

        # Recalculate bounding boxes
        re_bboxes_8  = distance2bbox(tiled_anchor_centers_8, bbox_8 * 8)
        re_bboxes_16 = distance2bbox(tiled_anchor_centers_16, bbox_16 * 16)
        re_bboxes_32 = distance2bbox(tiled_anchor_centers_32, bbox_32 * 32)

        # Recalculate keypoints
        re_kps_8  = distance2kps(tiled_anchor_centers_8, kps_8 * 8)
        re_kps_16 = distance2kps(tiled_anchor_centers_16, kps_16 * 16)
        re_kps_32 = distance2kps(tiled_anchor_centers_32, kps_32 * 32)

        # Recalculate score
        negative_score_8  = 1.0 - score_8
        re_score_8        = torch.cat([negative_score_8, score_8], dim = -1)
        negative_score_16 = 1.0 - score_16
        re_score_16       = torch.cat([negative_score_16, score_16], dim = -1)
        negative_score_32 = 1.0 - score_32
        re_score_32       = torch.cat([negative_score_32, score_32], dim = -1)

        # Concatenate all result
        re_bboxes = torch.cat([re_bboxes_8, re_bboxes_16, re_bboxes_32], dim = 1)
        re_bboxes = re_bboxes.unsqueeze(2)
        re_scores = torch.cat([re_score_8, re_score_16, re_score_32], dim = 1)
        re_kps = torch.cat([re_kps_8, re_kps_16, re_kps_32], dim = 1)
        re_kps = re_kps.unsqueeze(2)
        return re_bboxes, re_scores, re_kps

if __name__ == '__main__':
    model = SCRFDPostProcessor()
    model.eval()

    

    score_8  = torch.rand(size = (1, int(input_width // 8)*int(input_height // 8)*2, 1))
    score_16 = torch.rand(size = (1, int(input_width // 16)*int(input_height // 16)*2, 1))
    score_32 = torch.rand(size = (1, int(input_width // 32)*int(input_height // 32)*2, 1))
    bbox_8   = torch.rand(size = (1, int(input_width // 8)*int(input_height // 8)*2, 4))
    bbox_16  = torch.rand(size = (1, int(input_width // 16)*int(input_height // 16)*2, 4))
    bbox_32  = torch.rand(size = (1, int(input_width // 32)*int(input_height // 32)*2, 4))
    kps_8    = torch.rand(size = (1, int(input_width // 8)*int(input_height // 8)*2, 10))
    kps_16   = torch.rand(size = (1, int(input_width // 16)*int(input_height // 16)*2, 10))
    kps_32   = torch.rand(size = (1, int(input_width // 32)*int(input_height // 32)*2, 10))

    bboxes, scores, kps = model(score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32)
    print(bboxes.shape, scores.shape, kps.shape)
    print(scores)
    # Export the model
    torch.onnx.export(model,               # model being run
                      (score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32),                         # model input (or a tuple for multiple inputs)
                      "scrfd_post_process.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32'],   # the model's input names
                      output_names = ['bboxes', 'scores', 'kps'], # the model's output names
                      dynamic_axes={'score_8' : [0, 1],    # variable length axes
                                    'score_16' : [0, 1],    # variable length axes
                                    'score_32' : [0, 1],    # variable length axes
                                    'bbox_8' : [0, 1],    # variable length axes
                                    'bbox_16' : [0, 1],    # variable length axes
                                    'bbox_32' : [0, 1],    # variable length axes
                                    'kps_8' : [0, 1],    # variable length axes
                                    'kps_16' : [0, 1],    # variable length axes
                                    'kps_32' : [0, 1],    # variable length axes
                                    'bboxes' : [0, 1],
                                    'scores' : [0, 1],
                                    'kps' : [0, 1]})

    # import sclblonnx as so

    # sg1 = so.graph_from_file('weights/scrfd.onnx')
    # sg2 = so.graph_from_file('scrfd_post_process.onnx')
    
    # so.list_outputs(sg1)
    # so.list_inputs(sg2)

    # g = so.merge(sg1, sg2, outputs=["bboxes", "scores", "kps"], inputs=["input.1"])
    # so.check(g)
    # so.display(g)
    # g = so.graph_to_file(g, "weights/scrfd-nms.onnx")

    import onnx
    import onnx_graphsurgeon as gs
    model1 = onnx.load('scrfd.onnx')
    model1.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
    model1.graph.input[0].type.tensor_type.shape.dim[2].dim_param = str(input_height)
    model1.graph.input[0].type.tensor_type.shape.dim[3].dim_param = str(input_width)
    for i in range(len(model1.graph.node)):
        model1.graph.node[i].name = 'infer-' + model1.graph.node[i].name
    model2 = onnx.load('scrfd_post_process.onnx')
    for i in range(len(model2.graph.node)):
        model2.graph.node[i].name = 'post-' + model2.graph.node[i].name
    combined_model = onnx.compose.merge_models(
        model1, model2,
        io_map=[('score_8', 'score_8'),
                ('score_16', 'score_16'),
                ('score_32', 'score_32'),
                ('bbox_8', 'bbox_8'),
                ('bbox_16', 'bbox_16'),
                ('bbox_32', 'bbox_32'),
                ('kps_8', 'kps_8'),
                ('kps_16', 'kps_16'),
                ('kps_32', 'kps_32'),]
                )
    onnx.checker.check_model(combined_model)
    graph = gs.import_onnx(combined_model)

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "scrfd-post-{}-{}.onnx".format(input_width, input_height))

    