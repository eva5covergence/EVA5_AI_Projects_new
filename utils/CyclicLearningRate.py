'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import matplotlib.pyplot as plt
import numpy as np

class CyclicLR:
    def __init__(self,lr_max,lr_min,step_size, num_iterations):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.lr = []
        self.pad_factor = lr_max / 10
        
    def cycle(self,iteration):
        return np.floor(1+(iteration/(2*self.step_size)))
    
    def lr_position(self, iteration, cycle):
        return abs((iteration / self.step_size) - (2 * cycle) + 1)
    
    def current_lr(self,lr_position):
        return self.lr_min+(self.lr_max-self.lr_min)*(1-lr_position)
    
    def cyclic_lr(self, plot=True):
        for iteration in range(self.num_iterations):
            cyl = self.cycle(iteration)
            lr_pos = self.lr_position(iteration,cyl)
            cur_lr = self.current_lr(lr_pos)
            self.lr.append(cur_lr)
        
        if plot:
            self.plot()
    
    def plot(self):
        # Initialize a figure
        fig = plt.figure(figsize=(10, 3))
        
        #set plot title
        plt.title('Cyclic LR')
        
        # Label axes
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        
        #adding horizantal line for max lr
        plt.axhline(self.lr_max,0.03,0.97,label='max_lr', color='y')
        plt.text(0, self.lr_max + self.pad_factor, 'max_lr')
        
        #adding horizantal line for min lr
        plt.axhline(self.lr_min,0.03,0.97,label='min_lr', color='y')
        plt.text(0, self.lr_min - self.pad_factor, 'min_lr')
        
        # Plot lr change
        plt.plot(self.lr)
        
        # Plot margins and save plot
        plt.margins(y=0.2)
        plt.tight_layout()
        plt.savefig('clr_plot.png')
        
        
        



