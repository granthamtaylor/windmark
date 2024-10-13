# def collate(batch: list[InputTensor]) -> InputTensor:
#     return torch.stack(batch, dim=0)


# class BitMapIterator(torch.utils.data.IterableDataset):
#     def __init__(self, image: os.PathLike) -> None:
#         super().__init__()

#         with Image.open(image) as file:
#             self.data = np.array(file)

#         self.x, self.y, C = self.data.shape

#         if C != 3:
#             raise ValueError("image must have 3 channels (RGB)")

#     def __len__(self) -> int:
#         return self.x * self.y

#     def generate(self):
#         n_workers: int = torch.utils.data.get_worker_info().num_workers
#         worker: int = torch.utils.data.get_worker_info().id

#         coordinates = list(itertools.product(range(self.x), range(self.y)))

#         random.shuffle(coordinates)

#         for x, y in coordinates:
#             # calculate unique pixel index
#             index = x * self.x + y

#             if index % n_workers == worker:
#                 colors = torch.tensor(self.data[x, y], dtype=torch.int32).split(1, dim=0)

#                 yield InputTensor(
#                     coordinates=CoordinateTensor(torch.tensor([x]), torch.tensor([y])),
#                     color=PixelTensor(*colors),
#                 )

#     def __iter__(self):
#         return iter(self.generate())
