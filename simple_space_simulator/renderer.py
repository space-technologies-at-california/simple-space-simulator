import matplotlib.pyplot as plt

class Renderer:
    def __init__(self):
        self.time_stamps = []
        self.states = []
        self.plots = []
        
    def run(self, simulator, stop_condition):
        while not stop_condition(self.states, self.time_stamps):
            time, state = simulator.step()
            self.states.append(state)
            self.time_stamps.append(time)
            
    def add_plot(self, plot):
        self.plots.append(plot)
        
    def clear_plots(self):
        self.plots = []
                    
    def render(self, figsize=(7,5), columns=3):
        fig = plt.figure(figsize=figsize)
        for i, plot in enumerate(self.plots):
            if plot.is_3d:
                ax = fig.add_subplot((len(self.plots)+1)//columns if not columns == 1 else len(self.plots), columns, i+1, projection='3d')
            else:
                ax = fig.add_subplot((len(self.plots)+1)//columns if not columns == 1 else len(self.plots), columns, i+1)
            plot.build(self.states, self.time_stamps, ax)
        plt.show()