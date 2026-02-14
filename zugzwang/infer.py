import asyncio
from functools import partial

import torch

from .encode import board_to_tensor


class InferenceBatcher:
    def __init__(
        self, model, device, batch_size=64, timeout=0.005, history_steps=8
    ):
        self._model = model
        self._model.eval()
        self._model.to(device)
        self._device = device
        self._batch_size = batch_size
        self._timeout = timeout
        self.history_steps = history_steps
        self._queue = asyncio.Queue()

    async def infer(self, board):
        tensor = board_to_tensor(board, self.history_steps)
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((tensor, future))
        return await future

    def _forward(self, batch):
        with torch.no_grad():
            policy, value = self._model(batch)
        policy = torch.softmax(policy, dim=1).cpu()
        value = value.squeeze(-1).cpu()
        return policy, value

    async def run(self):
        loop = asyncio.get_event_loop()
        while True:
            # Block until at least one request arrives
            tensor, future = await self._queue.get()
            batch_tensors = [tensor]
            batch_futures = [future]

            # Collect more until batch_size or timeout
            deadline = loop.time() + self._timeout
            while len(batch_tensors) < self._batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    tensor, future = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch_tensors.append(tensor)
                    batch_futures.append(future)
                except asyncio.TimeoutError:
                    break

            # GPU forward pass in executor so game coroutines aren't blocked
            batch = torch.stack(batch_tensors).to(self._device)
            policy, value = await loop.run_in_executor(
                None, partial(self._forward, batch)
            )

            # Resolve futures so each coroutine gets its result
            for i, fut in enumerate(batch_futures):
                fut.set_result((policy[i], value[i].item()))
