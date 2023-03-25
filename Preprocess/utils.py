def path_to_save(path: str) -> str:
    split_path = path.split('/')
    file_name = split_path[-1]
    save_path = f'./processed/{file_name}'
    return save_path
