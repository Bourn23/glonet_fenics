import torch

v = torch.randn(10, requires_grad=True, dtype = torch.float64)
optimizer = torch.optim.Adam([v], lr = 1e4)
print(v)
p = (v**2 - 20) * 4

orig = torch.randn(10, dtype = torch.float64)

loss = torch.nn.MSELoss()
diff = loss(orig, v)
optimizer.zero_grad()
diff.backward()
optimizer.step()
print(v)

