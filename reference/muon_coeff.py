import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

torch.set_default_dtype(torch.float64)


def f_i(x, a, b, c):
    return a * x + b * x**3 + c * x**5

hh = 0.926
tt = 0.1539
yy = 0.97
zz = 0.0639
R = (torch.rand((15, 262144)) * zz + yy)
s0 = (torch.rand((262144,)) * tt + hh)
s1 = (torch.rand((262144,)) * tt + hh)
s2 = (torch.rand((262144,)) * tt + hh)
s3 = (torch.rand((262144,)) * tt + hh)
s4 = (torch.rand((262144,)) * tt + hh)

def iterate_f(x, params):
    a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, a5, b5, c5 = params.unsqueeze(1) * R
    y1 = f_i(x * s0, a1, b1, c1) * s1
    y2 = f_i(y1, a2, b2, c2) * s2
    y3 = f_i(y2, a3, b3, c3) * s3
    y4 = f_i(y3, a4, b4, c4) * s4
    y5 = f_i(y4, a5, b5, c5) 
    return y5

def loss_function(x, params):
    y = iterate_f(x, params)
    return ((y - 1)**2).mean()

params = torch.tensor(
    [
        3.9278, -8.8469,  5.3948,  3.4335, -5.5413,  2.4535,  3.5228, -5.4616,
        2.3485,  3.6798, -5.0828,  2.0279,  2.7327, -2.6189,  0.8640
    ], requires_grad=True)

optimizer = optim.AdamW([params], lr=0.0005, eps=1e-40, betas=(0.8, 0.9), weight_decay=0.0)

num_epochs = 10000
batch_size = 262144

with torch.no_grad():
    s = torch.randn((batch_size,))
    u = 10**(-1.9 + 1.1*s)
    x = torch.where((1e-4<u) & (u<1), u, torch.rand(batch_size)*(1-1e-4)+1e-4)

plt.ion()
fig, ax = plt.subplots()
x_vals = np.linspace(0, 1.1, 100)

for epoch in range(num_epochs):
    loss = loss_function(x, params)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, a5, b5, c5 = params.detach().numpy()
        
        y1_vals = a1 * x_vals + b1 * x_vals**3 + c1 * x_vals**5
        y2_vals = a2 * x_vals + b2 * x_vals**3 + c2 * x_vals**5
        y3_vals = a3 * x_vals + b3 * x_vals**3 + c3 * x_vals**5
        y4_vals = a4 * x_vals + b4 * x_vals**3 + c4 * x_vals**5
        y5_vals = a5 * x_vals + b5 * x_vals**3 + c5 * x_vals**5
        
        ax.clear()
        
        ax.plot(x_vals, y1_vals, label=f"$f_1(x)$")
        ax.plot(x_vals, y2_vals, label=f"$f_2(x)$")
        ax.plot(x_vals, y3_vals, label=f"$f_3(x)$")
        ax.plot(x_vals, y4_vals, label=f"$f_4(x)$")
        ax.plot(x_vals, y5_vals, label=f"$f_5(x)$")
        
        ax.set_title(f"Polynomials at Epoch {epoch + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)
        print(params)

        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()

print("Final Parameters:")
print(params)