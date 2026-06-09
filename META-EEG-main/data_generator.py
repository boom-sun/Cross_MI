import torch
import random

def task_generator(config, data, label, s_label, iteration, mini_task_size, ti, Subject):
    # 1️⃣，先拷贝一份，不要在原 Subject 上做 in-place 改动
    all_subjects = list(Subject)

    # 2️⃣，从可选的 subject 中去掉当前 ti（当前 ti 通常是留给主循环外层用的）
    available_subjects = [sid for sid in all_subjects if sid != ti]

    # 3️⃣，从 config 里读“每次选多少个 subject 当 query”
    num_query = config['train'].get('num_query_subject', len(available_subjects))

    # 保证不会超过实际可用的 subject 数
    if num_query < len(available_subjects):
        # 随机采样 num_query 个 subject 当 query
        query_subjects = random.sample(available_subjects, num_query)
    else:
        # 如果设得比 36 还大，就等价于全用
        query_subjects = available_subjects

    tasks_data = []
    tasks_labels = []

    # 4️⃣，只对“被选中的这些 subject”构造任务
    for task_idx in query_subjects:
        # 这个 subject 的数据做 query
        query_set_data = data[s_label == task_idx]
        query_set_labels = label[s_label == task_idx]

        # 其余所有 subject 的数据（包括 ti 在内）做 support
        support_mask = (s_label != task_idx)
        support_set_data = data[support_mask]
        support_set_labels = label[support_mask]

        # 打乱 support
        indices = torch.randperm(support_set_data.size(0))
        shuffled_spt_data = support_set_data[indices]
        shuffled_spt_labels = support_set_labels[indices]

        task_data = (shuffled_spt_data, query_set_data)
        task_labels = (shuffled_spt_labels, query_set_labels)
        tasks_data.append(task_data)
        tasks_labels.append(task_labels)

    # 5️⃣ 保持你原来的 split_data + mini_task 逻辑不变
    def split_data(data_tensor, chunk_size):
        result = []
        current_idx = 0
        while current_idx < len(data_tensor):
            result.append(data_tensor[current_idx:current_idx + chunk_size])
            current_idx += chunk_size
        return result

    mini_tasks_data = [
        (spt, task[1])
        for task in tasks_data
        for spt in split_data(task[0], mini_task_size)
    ]
    mini_tasks_label = [
        (spt, task[1])
        for task in tasks_labels
        for spt in split_data(task[0], mini_task_size)
    ]

    if iteration == 0:
        print("Task 1 - Support Set Data:")
        print(mini_tasks_data[0][0].shape)
        print("Task 1 - Query Set Data:")
        print(mini_tasks_data[0][1].shape)
        print("Task 1 - Support Set Labels:")
        print(mini_tasks_label[0][0].shape)
        print("Task 1 - Query Set Labels:")
        print(mini_tasks_label[0][1].shape)

    return [mini_tasks_data, mini_tasks_label]
