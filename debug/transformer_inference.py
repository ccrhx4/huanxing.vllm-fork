from transformers import AutoTokenizer
import torch

device = "cuda"

if device == "hpu":
    import habana_frameworks.torch as ht

model_name = "Qwen/Qwen2-7B-Instruct"
input_tokens = [785, 2701, 525, 5248, 5754, 4755, 320, 4197, 11253, 8, 911, 29803, 13, 21149, 3019, 553, 3019, 323, 1221, 6248, 697, 4226, 448, 330, 1782, 4226, 374, 320, 55, 9940, 1380, 1599, 374, 279, 4396, 6524, 5754, 624, 14582, 510, 53544, 279, 1372, 315, 5128, 304, 279, 468, 6480, 19745, 315, 264, 6291, 315, 220, 16, 18, 34, 31801, 78474, 17855, 320, 16, 18, 2149, 18, 6667, 701, 25538, 279, 5128, 653, 537, 27248, 624, 3798, 510, 32, 13, 220, 16, 15, 198, 33, 13, 220, 23, 198, 34, 13, 220, 19, 198, 35, 13, 220, 17, 15, 198, 36, 13, 220, 16, 17, 198, 37, 13, 220, 18, 198, 38, 13, 220, 16, 21, 198, 39, 13, 220, 20, 198, 40, 13, 220, 17, 19, 198, 41, 13, 220, 21, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 576, 16715, 1685, 38000, 56981, 19745, 686, 387, 6718, 553, 1378, 7586, 315, 21880, 13, 576, 1156, 374, 279, 17071, 62057, 16230, 448, 279, 220, 16, 18, 34, 320, 77, 9637, 12616, 400, 40, 284, 715, 59, 37018, 90, 16, 15170, 17, 31716, 8, 892, 686, 6718, 279, 19745, 1119, 220, 17, 5128, 13, 1096, 686, 387, 4623, 6718, 1119, 220, 19, 5128, 553, 279, 16230, 448, 2326, 13578, 220, 16, 39, 96092, 13, 576, 2790, 1372, 315, 5128, 374, 8916, 400, 17, 1124, 50853, 220, 19, 284, 220, 23, 12947, 576, 4226, 374, 320, 33, 3593, 14582, 510, 23085, 315, 279, 2701, 11469, 279, 6275, 67, 18245, 315, 1874, 12, 16, 19, 5424, 304, 1973, 315, 28387, 19753, 11, 504, 15457, 311, 8426, 5267, 3798, 510, 32, 13, 4229, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 366, 11832, 39, 19, 366, 6826, 19, 198, 33, 13, 11832, 39, 19, 366, 4229, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 366, 6826, 19, 198, 34, 13, 97354, 39, 19, 366, 6826, 19, 366, 13059, 39, 19, 366, 4229, 39, 19, 366, 11832, 39, 19, 198, 35, 13, 97354, 39, 19, 366, 13059, 39, 19, 366, 6826, 19, 366, 4229, 39, 19, 366, 11832, 39, 19, 198, 36, 13, 13059, 39, 19, 366, 4229, 39, 19, 366, 11832, 39, 19, 366, 97354, 39, 19, 366, 6826, 19, 198, 37, 13, 6826, 19, 366, 4229, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 366, 11832, 39, 19, 198, 38, 13, 11832, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 366, 4229, 39, 19, 366, 6826, 19, 198, 39, 13, 6826, 19, 366, 11832, 39, 19, 366, 4229, 39, 19, 366, 13059, 39, 19, 366, 97354, 39, 19, 198, 40, 13, 6826, 19, 366, 97354, 39, 19, 366, 4229, 39, 19, 366, 13059, 39, 19, 366, 11832, 39, 19, 198, 41, 13, 97354, 39, 19, 366, 13059, 39, 19, 366, 4229, 39, 19, 366, 11832, 39, 19, 366, 6826, 19, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 576, 28387, 19753, 315, 1874, 12, 16, 19, 6275, 67, 18245, 42054, 438, 582, 3271, 504, 279, 1909, 315, 1874, 220, 16, 19, 311, 279, 5622, 13, 576, 1973, 315, 5424, 304, 279, 1874, 504, 1909, 311, 5622, 374, 356, 11, 11832, 11, 4229, 11, 13059, 11, 97354, 13, 15277, 304, 1973, 315, 7703, 28387, 19753, 582, 614, 97354, 39, 19, 11, 13059, 39, 19, 11, 4229, 39, 19, 11, 11832, 39, 19, 11, 323, 6826, 19, 11, 476, 4226, 320, 41, 568, 576, 4226, 374, 320, 41, 3593, 14582, 510, 23085, 315, 279, 2701, 374, 6509, 458, 13621, 458, 8503, 67, 1399, 5267, 3798, 510, 32, 13, 472, 17, 13880, 18, 198, 33, 13, 12812, 5066, 198, 34, 13, 6826, 19, 198, 35, 13, 472, 8996, 18, 198, 36, 13, 5627, 17, 198, 37, 13, 1674, 77927, 18, 8, 18, 198, 38, 13, 14413, 8281, 18, 198, 39, 13, 472, 17, 46, 198, 40, 13, 472, 5066, 198, 41, 13, 451, 10360, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 1527, 13621, 458, 8503, 67, 1399, 374, 264, 23628, 429, 374, 14257, 553, 17592, 3015, 504, 458, 13621, 13, 576, 11483, 14806, 369, 3015, 374, 472, 17, 46, 11, 892, 3363, 429, 582, 1184, 311, 8253, 892, 315, 1493, 2606, 11, 979, 10856, 448, 472, 17, 46, 11, 7586, 458, 13621, 13, 5627, 17, 11, 476, 328, 14308, 324, 39489, 11, 979, 10856, 448, 472, 17, 46, 11, 3643, 472, 17, 13880, 19, 11, 476, 71491, 292, 13621, 13, 576, 4226, 374, 320, 36, 3593, 14582, 510, 32, 501, 23628, 374, 91006, 323, 1730, 311, 387, 264, 1615, 453, 4640, 292, 13621, 448, 264, 296, 7417, 3072, 315, 220, 17, 19, 23, 342, 38871, 13, 3197, 220, 15, 13, 15, 15, 20, 15, 21609, 315, 419, 13621, 525, 55667, 304, 220, 15, 13, 20, 15, 15, 444, 315, 3015, 11, 279, 36043, 374, 16878, 438, 220, 18, 13, 23, 24, 13, 3555, 374, 279, 281, 82968, 315, 419, 13621, 5267, 3798, 510, 32, 13, 220, 20, 13, 22, 23, 198, 33, 13, 220, 19, 13, 22, 23, 198, 34, 13, 220, 19, 13, 20, 21, 198, 35, 13, 220, 21, 13, 23, 24, 198, 36, 13, 220, 22, 13, 22, 23, 198, 37, 13, 220, 18, 13, 23, 24, 198, 38, 13, 220, 16, 13, 17, 18, 198, 39, 13, 220, 17, 13, 23, 24, 198, 40, 13, 220, 17, 13, 18, 18, 198, 41, 13, 220, 20, 13, 18, 18, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 79540, 429, 400, 58, 32, 60, 284, 508, 39, 47822, 10, 25439, 12947, 5692, 11, 419, 374, 6144, 311, 26107, 16, 15, 87210, 18, 13, 23, 24, 92, 12947, 5005, 582, 614, 400, 42, 15159, 64, 92, 284, 24437, 59, 37018, 90, 58, 39, 47822, 10, 92, 1457, 32, 87210, 92, 13989, 90, 58, 17020, 13989, 284, 715, 59, 37018, 90, 16, 15, 87210, 18, 13, 23, 24, 92, 1124, 50853, 220, 16, 15, 87210, 18, 13, 23, 24, 3417, 90, 16, 15, 87210, 17, 3417, 13, 576, 12942, 27690, 374, 400, 12, 18, 13, 23, 24, 488, 10293, 18, 13, 23, 24, 8, 481, 10293, 17, 8, 284, 220, 20, 13, 22, 23, 54876, 8916, 400, 42, 4306, 284, 220, 16, 15, 87210, 20, 13, 22, 23, 92, 12947, 576, 400, 79, 42, 4306, 3, 374, 279, 8225, 1487, 315, 400, 42, 4306, 54876, 892, 374, 6144, 311, 400, 20, 13, 22, 23, 12947, 576, 4226, 374, 320, 32, 3593, 14582, 510, 32, 6291, 5610, 220, 17, 13, 15, 15, 34651, 315, 1613, 5298, 13621, 11, 6826, 18, 8281, 46761, 11, 323, 220, 16, 13, 15, 15, 34651, 315, 34619, 64702, 349, 11, 14413, 82934, 18, 8281, 46, 8, 17, 13, 576, 6291, 374, 2952, 311, 22106, 279, 5256, 315, 264, 2613, 3311, 315, 3746, 13621, 476, 3746, 2331, 448, 1172, 8922, 4344, 304, 279, 36043, 315, 279, 6291, 13, 80808, 32676, 315, 3746, 13621, 476, 3746, 2331, 646, 5240, 264, 5089, 2297, 304, 36043, 13, 2585, 1657, 4544, 642, 315, 24691, 2216, 13621, 11, 472, 8996, 18, 11, 1231, 387, 3694, 1573, 279, 36043, 12033, 311, 2297, 11941, 5267, 3798, 510, 32, 13, 220, 15, 13, 17, 20, 15, 34651, 198, 33, 13, 220, 15, 13, 20, 15, 15, 34651, 198, 34, 13, 220, 18, 13, 15, 15, 34651, 198, 35, 13, 220, 16, 13, 15, 15, 34651, 198, 36, 13, 220, 18, 13, 20, 15, 34651, 198, 37, 13, 220, 16, 13, 20, 15, 34651, 198, 38, 13, 220, 17, 13, 20, 15, 34651, 198, 39, 13, 220, 19, 13, 15, 15, 34651, 198, 40, 13, 220, 15, 13, 22, 20, 15, 34651, 198, 41, 13, 220, 17, 13, 15, 15, 34651, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13, 1205, 1035, 1075, 311, 12564, 279, 4147, 8654, 315, 419, 6291, 13, 5512, 582, 3270, 279, 23606, 369, 279, 27672, 2022, 315, 279, 7469, 13621, 11, 304, 419, 1142, 315, 1613, 5298, 13621, 13, 400, 2149, 15159, 18, 92, 8281, 46761, 320, 36306, 8, 488, 472, 15159, 17, 92, 46, 715, 491, 6044, 472, 15159, 18, 92, 46, 47822, 10, 92, 488, 6826, 18, 8281, 46, 87210, 92, 12947, 576, 63280, 349, 2331, 374, 8916, 279, 64702, 349, 27672, 13, 576, 3694, 3746, 13621, 11, 49516, 2216, 13621, 11, 686, 13767, 448, 279, 63280, 349, 2331, 13, 15277, 279, 7192, 3311, 315, 13621, 429, 646, 387, 3694, 686, 387, 6144, 311, 279, 3311, 315, 64702, 349, 27672, 11, 476, 220, 17, 4544, 642, 13, 576, 4226, 374, 320, 41, 3593, 14582, 510, 4340, 1657, 2544, 3664, 388, 315, 220, 15, 13, 17, 20, 15, 386, 37512, 39, 1558, 432, 1896, 311, 20628, 551, 6587, 220, 20, 15, 13, 15, 64070, 315, 220, 15, 13, 16, 20, 15, 386, 472, 18, 2045, 19, 5267, 3798, 510, 32, 13, 220, 22, 20, 13, 15, 64070, 198, 33, 13, 220, 24, 15, 13, 15, 64070, 198, 34, 13, 220, 21, 15, 13, 15, 64070, 198, 35, 13, 220, 16, 17, 15, 64070, 198, 36, 13, 220, 18, 15, 13, 15, 64070, 198, 37, 13, 220, 16, 23, 15, 64070, 198, 38, 13, 220, 17, 22, 15, 64070, 198, 39, 13, 220, 16, 15, 15, 64070, 198, 40, 13, 220, 17, 22, 64070, 198, 41, 13, 220, 16, 20, 15, 64070, 198, 16141, 25, 6771, 594, 1744, 3019, 553, 3019, 13]

tokens = []
tokens.append(input_tokens)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto').to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

seq = torch.LongTensor(tokens).to(device)
generated_ids = model.generate(seq, max_length=2048, top_k=0, do_sample=False, repetition_penalty=1.0, top_p=1.0, temperature=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
