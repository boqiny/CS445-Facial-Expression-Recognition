import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def compute_iou(box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou

class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = x - 0.5 * w
        y1 = y - 0.5 * h
        x2 = x + 0.5 * w
        y2 = y + 0.5 * h

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        box_target_xyxy = self.xywh2xyxy(box_target)
        best_iou = torch.zeros((box_target_xyxy.size(0), 1), device=box_target.device)
        best_boxes = torch.zeros((box_target_xyxy.size(0), 5), device=box_target.device)

        for pred_boxes in pred_box_list:
            pred_boxes_xyxy = self.xywh2xyxy(pred_boxes[:, :4])
            confidences = pred_boxes[:, 4] 
            ious = compute_iou(box_target_xyxy, pred_boxes_xyxy)
            max_iou, max_indices = ious.max(dim=1, keepdim=True)
            update_mask = max_iou.squeeze(-1) > best_iou.squeeze(-1)
            best_iou[update_mask] = max_iou[update_mask]
            best_boxes[update_mask, :4] = pred_boxes_xyxy[max_indices.squeeze(-1), :][update_mask]
            best_boxes[update_mask, 4] = confidences[max_indices.squeeze(-1)][update_mask]


        return best_iou, best_boxes

    
    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        error = classes_target - classes_pred
        error = (error * error) * has_object_map.unsqueeze(-1).expand_as(error) 
        loss = torch.sum(error)  

        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        no_object_loss = 0.0
        for pred_boxes in pred_boxes_list:
            confidence_scores = pred_boxes[..., 4] 
            no_object_mask = (has_object_map == 0).float() 
            no_object_confidences = confidence_scores * no_object_mask
            loss = torch.sum((no_object_confidences ** 2))
            no_object_loss += loss
        
        num_cells = no_object_mask.numel() * len(pred_boxes_list) 
        no_object_loss /= num_cells
        
        return no_object_loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        GT = box_target_conf.detach()
        error = box_pred_conf - GT
        error = error * error
        loss = torch.sum(error) * self.l_coord 

        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        error_x = box_pred_response[:, 0] - box_target_response[:, 0]
        error_y = box_pred_response[:, 1] - box_target_response[:, 1]
        error_w = (box_pred_response[:, 2])**0.5 - (box_target_response[:, 2])**0.5
        error_h = (box_pred_response[:, 3])**0.5 - (box_target_response[:, 3])**0.5

        loss_coord = torch.sum((error_x*error_x) + (error_y*error_y))
        loss_size = torch.sum((error_w*error_w) + (error_h*error_h))
        reg_loss = self.l_coord * (loss_size + loss_coord)  

        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        N = pred_tensor.size(0)
        total_loss = 0.0
        pred_boxes_tensor = pred_tensor[:, :, :, 0:(self.B*5)]
        pred_boxes_list = []
        for i in range(0, (self.B*5), 5):
            pred_boxes_list.append(pred_boxes_tensor[:, :, :, i:(i+5)])  
        pred_cls = pred_tensor[:, :, :, (self.B*5):pred_tensor.size(3)]  
        cls_loss = (self.get_class_prediction_loss(pred_cls, target_cls, has_object_map))/N
        no_obj_loss = (self.get_no_object_loss(pred_boxes_list, has_object_map))/N
        for i in range(len(pred_boxes_list)):
            box = pred_boxes_list[i]
            box = box[has_object_map.unsqueeze(-1).expand_as(box) == True]
            pred_boxes_list[i] = box.reshape((-1, 5)) 
        target_boxes = target_boxes[has_object_map.unsqueeze(-1).expand_as(target_boxes) == True]
        target_boxes = target_boxes.reshape((-1, 4))
        best_iou, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)
        reg_loss = (self.get_regression_loss(best_boxes[:, 0:4], target_boxes))/N 
        containing_obj_loss = (self.get_contain_conf_loss(best_boxes[:, 4], best_iou))/N
        total_loss = reg_loss + containing_obj_loss + no_obj_loss + cls_loss
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=containing_obj_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss
        )
        return loss_dict
    
def mock_data():
    # Mock predictions and target tensors mimic real scenario
    N, S, B = 1, 7, 2  # Batch size, grid size, number of boxes per grid cell
    C = 20  # Number of classes
    pred_tensor = torch.rand(N, S, S, B*5+C)  # Random predictions
    target_boxes = torch.rand(N, S, S, 4)  # Random target boxes
    target_cls = torch.randint(0, 2, (N, S, S, C)).float()  # Random target class labels
    has_object_map = torch.randint(0, 2, (N, S, S)).bool()  # Random presence of objects in cells

    return pred_tensor, target_boxes, target_cls, has_object_map

def test_loss():
    pred_tensor, target_boxes, target_cls, has_object_map = mock_data()
    loss_module = YoloLoss(S=7, B=2, l_coord=5, l_noobj=0.5)
    loss_dict = loss_module(pred_tensor, target_boxes, target_cls, has_object_map)
    print("Loss Dictionary:", loss_dict)

test_loss()