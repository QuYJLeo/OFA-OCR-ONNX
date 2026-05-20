import os, time
import numpy as np
import onnxruntime as ort
from utils.preprocess import Preprocessor
from utils.tokenization import TokenizerZH
ort.set_default_logger_severity(3)

def log_softmax(x, axis=-1):
    # 计算 softmax
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    # 计算 log softmax
    log_softmax_x = np.log(softmax_x)

    return log_softmax_x


class OfaTasks():
    def __init__(self):
        self.tokenizer = TokenizerZH()

        self.preprocessor = Preprocessor()

        # generator
        self.encoder_session = ort.InferenceSession("./onxw/encoder_fp16.onnx")
        self.decoder_session = ort.InferenceSession("./onxw/decoder_fp16.onnx")

    def __call__(self, input):
        patch_images = self.preprocessor(input)
        patch_images = patch_images.astype(np.float16)
        print("==========================encode, ", patch_images.shape,  patch_images.dtype)
        t1 = time.time()
        outputs = self.encoder_session.run(None,
                                           {"patch_images": patch_images})  # [1, 3, 480, 480]
        t2 = time.time()
        print("encoder time: ", t2 - t1)
        last_hidden_state = outputs[0]  # (5, 912, 768)  np.float16
        pad_mask = outputs[1]  # (5, 912)  np.bool
        pos_embed = outputs[15]  # (5, 912, 768)  np.float16



        print("==========================decode==========================")
        attn = np.empty((5, 912, 18), dtype=np.float16)
        finalized = []
        reorder_state = None
        select = np.array([0, 0, 0, 0, 0])
        mask = np.array([[False, False, False, False, False]])


        # initialize buffers
        scores = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])


        tokens = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        active_hypos = np.array([[0, 1, 2, 3, 4]])

        t1 = time.time()
        for step in range(17):  # one extra step for EOS marker
            if reorder_state is not None:
                last_hidden_state = last_hidden_state[select]
                pad_mask = pad_mask[select]
                pos_embed = pos_embed[select]


            input_ids = tokens[:, :step + 1]
            encoder_att_mask = pad_mask[:, np.newaxis, np.newaxis, :]
            encoder_att_mask = np.broadcast_to(encoder_att_mask, (5, 1, step + 1, 912)).astype(np.float16)  # 数组广播到指定的形状
            encoder_att_mask = np.where(encoder_att_mask != 0, -np.inf, encoder_att_mask)  # 所有非零元素替换为负无穷大（-inf）

            inputs = {
                "input_ids": input_ids.astype(np.int64),  # ([5, dynamic])  np.int64
                "encoder_attention_mask": encoder_att_mask,  # ([5, 1, dynamic, 912]) np.float32
                "encoder_hidden_states": last_hidden_state,  # ([5, 912, 768]) np.float32
                "src_pos_embed": pos_embed,  # ([5, 912, 768]) np.float32
            }
            de_hid_states, de_cross_att = self.decoder_session.run(["hidden_states", "all_cross_attentions"], inputs)


            """
            注： de_hid_states.shape (5, step+1, 30325)  
                de_cross_att.shape    (5, 12, step+1, 912)
            """
            logits = de_hid_states[:, -1:, :]  # 提取最后一个时间步的隐藏状
            lprobs = log_softmax(logits, axis=-1)[:, -1, :]
            # print("lprobs:", lprobs.shape, lprobs.dtype)  # (5, 30325) float32

            if step == 0:
                lprobs[:, 2] = -np.inf

            lprobs[np.isnan(lprobs)] = -np.inf
            lprobs[:, 1] = -np.inf
            lprobs[:, 3] -= 0.0

            # handle max length constraint
            if step >= 16:
                lprobs[:, :2] = -np.inf
                lprobs[:, :2] = -np.inf


            avg_attn_scores = np.transpose(de_cross_att, (1, 0, 2, 3))   # (5, 12, dynamic, 912) --> (12, 5, dynamic, 912)
            avg_attn_scores = np.mean(avg_attn_scores, axis=0)  # 交换维度 ([12, 5, dynamic, 912])  求dim均值
            avg_attn_scores = avg_attn_scores[:, -1, :]  # [5, 912]
            attn[:, :, step + 1] = avg_attn_scores

            if step > 0:
                sc = scores[:, :step]  # (5, x) x:从0 - step
                sc = sc[:, step - 1]
                sc = np.expand_dims(sc, axis=-1) # (5, 1)
                lprobs = lprobs + sc


            # 展平为一维数组并找出前 k 个最大值的索引和值  aaaa的维度(5, 30325)
            flattened = lprobs.reshape(-1,)   # (151625,)

            top_indices = np.argpartition(flattened, -10)[-10:]  # 找到前10大的元素索引（无序）
            top_indices = top_indices[np.argsort(-flattened[top_indices])]  # 按降序排列这些索引  (10,)

            cand_scores = flattened[top_indices]  # 获取对应的得分
            cand_scores = np.expand_dims(cand_scores, axis=0)  #  (1, 10) float32

            # 计算 beam 编号（整除）和实际 token 索引（取余）
            cand_beams = np.floor_divide(top_indices, 30325)  # 整除，等价于 torch.div(..., rounding_mode='floor')
            cand_beams = np.expand_dims(cand_beams, axis=0)  # (1, 10) int64


            cand_indices = np.mod(top_indices, 30325)  # 取模，等价于 torch.fmod
            cand_indices = np.expand_dims(cand_indices, axis=0)  # (1, 10) int64


            eos_mask = (cand_indices == 2) & (np.isinf(cand_scores) == False)  # (1, 10)  bool  判断是否等于 2，并且得分不是 -inf

            # 对前5列应用掩码，将满足条件的位置置为 0
            eos_mask[:, :5][mask] = 0
            eos_bbsz_idx = np.where(eos_mask[:, :5].flatten())[0]
            # print("eos_bbsz_idx:", eos_bbsz_idx, eos_bbsz_idx.shape)


            # 如果有满足条件的索引
            if eos_bbsz_idx.size > 0:
                # 提取对应的 tokens 并跳过第一个索引
                tokens_clone = tokens[eos_bbsz_idx][:, 1:step + 2]
                # 设置最后一个位置为 2
                tokens_clone[:, step] = 2
                # 提取对应的得分并归一化
                eos_scores = cand_scores[:, :5].flatten()[eos_bbsz_idx]
                eos_scores /= (step + 1)
                # 添加到 finalized 列表并跳出循环

                finalized.append({'tokens': tokens_clone[0], 'score': eos_scores[0]})
                break

            # 使用 np.take 沿着 axis=1 获取指定列
            active_bbsz_idx = np.take(cand_beams, indices=active_hypos, axis=1).flatten()

            tokens[:, :step + 1] = tokens[active_bbsz_idx, :step + 1]
            tokens[:, step + 1] = np.take_along_axis(cand_indices, active_hypos, axis=1).flatten()


            if step > 0:
                scores[:, :step] = scores[active_bbsz_idx, :step]

            gathered = np.take_along_axis(cand_scores, active_hypos, axis=1)
            scores[:, step:step + 1] = gathered.reshape(-1, 1)


            attn[:, :, :step + 2] = attn[active_bbsz_idx, :, :step + 2]
            attn[:, :, :step + 2] = attn[active_bbsz_idx][:, :, :step + 2]

            reorder_state = active_bbsz_idx
        t2 = time.time()
        print("decoder time:", t2 - t1)
        # print("finalized:", finalized)
        out = self.postprocess(finalized)
        return out

    def postprocess(self, gen_outputs):
        # print([gen_outputs[0]['tokens']])
        result = self.tokenizer.bdecode([gen_outputs[0]['tokens']])
        result = [t.replace(' ', '') for t in result]
        return result


if __name__ == '__main__':
    import random
    root = os.path.join(os.getcwd(), "te")
    lst = os.listdir(root)
    random.shuffle(lst)
    # print("lst:", lst)
    imgs_path = lst[:]
    tk = OfaTasks()

    for img_path in imgs_path:
        print("被测图片: ", img_path)
        result = tk(os.path.join(root, img_path))
        print("result: ", result)
        print()
