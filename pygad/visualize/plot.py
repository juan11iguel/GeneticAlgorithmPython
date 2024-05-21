"""
The pygad.visualize.plot module has methods to create plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygad
from typing import Literal
from pathlib import Path
import inspect


class Plot:

    SupportedPlotTypes: list[str] = ["plot", "scatter", "bar"]

    def __init__():
        pass

    def check_req_generations_completed(self, ) -> None:

        # Get caller function name
        caller = inspect.stack()[1].function

        if self.generations_completed < 1:
            msg = f"The {caller} method can only be called after completing at least 1 generation but ({self.generations_completed}) is completed."
            self.logger.error(msg)

            raise RuntimeError(msg)

    def check_correct_plot_type(self, input_plot_type: str) -> None:

        if input_plot_type not in self.SupportedPlotTypes:
            msg = f"The plot_type parameter must be {self.SupportedPlotTypes} but {input_plot_type} found."
            self.logger.error(msg)

            raise ValueError(msg)

    def save_figure(self, save_dir: Path | str):
        plt.savefig(fname=Path(save_dir).with_suffix('.png'), bbox_inches='tight')
        self.logger.info(f"Figure saved to {save_dir}")

    def plot_fitness(self,
                     title: str = "Fitness evolution over generations",
                     xlabel: str = "Generation",
                     ylabel: str = "Fitness",
                     linewidth: int = 3,
                     font_size: int = 14,
                     plot_type: Literal["plot", "scatter", "bar"] = "plot",
                     color: str = "#64f20c",
                     label: str = None,
                     save_dir: Path | str = None,
                     show_plot: bool = True,
                     ax:plt.Axes = None):

        """
        Creates, shows, and returns a figure that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14. Can be a list/tuple/numpy.ndarray if the problem is multi-objective optimization.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            color: Color of the plot which defaults to "#64f20c". Can be a list/tuple/numpy.ndarray if the problem is multi-objective optimization.
            label: The label used for the legend in the figures of multi-objective problems. It is not used for single-objective problems.
            save_dir: Directory to save the figure.

        Returns the figure.
        """

        # Initial checks
        self.check_req_generations_completed()
        self.check_correct_plot_type(plot_type)

        if ax is None:
            fig, ax = plt.subplots()

        if type(self.best_solutions_fitness[0]) in [list, tuple, np.ndarray] and len(self.best_solutions_fitness[0]) > 1:
            # Multi-objective optimization problem.
            if type(linewidth) in pygad.GA.supported_int_float_types:
                linewidth = [linewidth]
                linewidth.extend([linewidth[0]]*len(self.best_solutions_fitness[0]))
            elif type(linewidth) in [list, tuple, np.ndarray]:
                pass

            if type(color) is str:
                color = [color]
                color.extend([None]*len(self.best_solutions_fitness[0]))
            elif type(color) in [list, tuple, np.ndarray]:
                pass
            
            if label is None:
                label = [None]*len(self.best_solutions_fitness[0])

            # Loop through each objective to plot its fitness.
            for objective_idx in range(len(self.best_solutions_fitness[0])):
                # Return the color, line width, and label of the current plot.
                current_color = color[objective_idx]
                current_linewidth = linewidth[objective_idx]
                current_label = label[objective_idx]
                # Return the fitness values for the current objective function across all best solutions acorss all generations.
                fitness = np.array(self.best_solutions_fitness)[:, objective_idx]

                if plot_type == "plot":
                    ax.plot(fitness,
                            linewidth=current_linewidth,
                            color=current_color,
                            label=current_label)

                elif plot_type == "scatter":
                    ax.scatter(range(len(fitness)),
                              fitness,
                              linewidth=current_linewidth,
                              color=current_color,
                              label=current_label)

                elif plot_type == "bar":
                    ax.bar(range(len(fitness)),
                          fitness,
                          linewidth=current_linewidth,
                          color=current_color,
                          label=current_label)
        else:
            # Single-objective optimization problem.
            if plot_type == "plot":
                ax.plot(self.best_solutions_fitness,
                                       linewidth=linewidth, 
                                       color=color, label=label)
            elif plot_type == "scatter":
                ax.scatter(range(len(self.best_solutions_fitness)),
                                          self.best_solutions_fitness, 
                                          linewidth=linewidth, 
                                          color=color, label=label)
            elif plot_type == "bar":
                ax.bar(range(len(self.best_solutions_fitness)),
                                      self.best_solutions_fitness, 
                                      linewidth=linewidth, 
                                      color=color, label=label)

        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)

        if label is not None:
            ax.legend(loc='upper left')

        if save_dir is not None:
            self.save_figure(save_dir)

        if show_plot:
            plt.show()

        return fig if ax is None else ax.figure

    def plot_new_solution_rate(self,
                               title: str = "New Solution Rate over generations",
                               xlabel: str = "Generation",
                               ylabel: str = "New Solution Rate",
                               linewidth: int = 3,
                               font_size: int = 14,
                               plot_type: Literal["plot", "scatter", "bar"] = "plot",
                               color: str = "#64f20c",
                               label: str = None,
                               save_dir: Path | str = None,
                               show_plot: bool = True,
                               ax: plt.Axes = None):

        """
        Creates, shows, and returns a figure that summarizes the rate of exploring new solutions. This method works only when save_solutions=True in the constructor of the pygad.GA class.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            color: Color of the plot which defaults to "#64f20c".
            save_dir: Directory to save the figure.

        Returns the figure.
        """

        # Initial checks
        self.check_req_generations_completed()
        self.check_correct_plot_type(plot_type)

        if self.save_solutions == False:
            self.logger.error("The plot_new_solution_rate() method works only when save_solutions=True in the constructor of the pygad.GA class.")
            raise RuntimeError("The plot_new_solution_rate() method works only when save_solutions=True in the constructor of the pygad.GA class.")

        if ax is None:
            fig, ax = plt.subplots()

        unique_solutions = set()
        num_unique_solutions_per_generation = []
        for generation_idx in range(self.generations_completed):
            
            len_before = len(unique_solutions)

            start = generation_idx * self.sol_per_pop
            end = start + self.sol_per_pop
        
            for sol in self.solutions[start:end]:
                unique_solutions.add(tuple(sol))
        
            len_after = len(unique_solutions)
        
            generation_num_unique_solutions = len_after - len_before
            num_unique_solutions_per_generation.append(generation_num_unique_solutions)

        if plot_type == "plot":
            ax.plot(num_unique_solutions_per_generation, linewidth=linewidth, color=color, label=label)
        elif plot_type == "scatter":
            ax.scatter(range(self.generations_completed), num_unique_solutions_per_generation, linewidth=linewidth,
                       color=color, label=label)
        elif plot_type == "bar":
            ax.bar(range(self.generations_completed), num_unique_solutions_per_generation, linewidth=linewidth,
                   color=color, label=label)

        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)

        if label is not None:
            ax.legend(loc='upper left')

        if save_dir is not None:
            self.save_figure(save_dir)

        if show_plot:
            plt.show()

        return fig if ax is None else ax.figure

    def plot_genes(self, 
                   title: str = "Chromosome evolution over generations",
                   xlabel: str = "Gene",
                   ylabel: str = "Value",
                   linewidth: float = 3.0,
                   font_size: float = 14.0,
                   plot_type: Literal["plot", "scatter", "bar"] = "plot",
                   graph_type: Literal["plot", "boxplot", "histogram"] ="plot",
                   figure_layout: Literal["compact", "vertical_distribution", "horizontal_distribution"] = "compact",
                   fill_color: str = "#64f20c",
                   color: str = "black",
                   solutions: Literal["all", "best"] = "all",
                   save_dir: str | Path = None,
                   show_plot: bool = True,
                   axs: plt.Axes = None):

        """
        Creates, shows, and returns a figure with number of subplots equal to the number of genes. Each subplot shows the gene value for each generation. 
        This method works only when save_solutions=True in the constructor of the pygad.GA class. 
        It also works only after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            graph_type: Type of the graph which can be either "plot" (default), "boxplot", or "histogram".
            fill_color: Fill color of the graph which defaults to "#64f20c". This has no effect if graph_type="plot".
            color: Color of the plot which defaults to "black".
            solutions: Defaults to "all" which means use all solutions. If "best" then only the best solutions are used.
            save_dir: Directory to save the figure.

        Returns the figure.
        """

        # Initial checks
        self.check_req_generations_completed()
        self.check_correct_plot_type(plot_type)

        if solutions not in ["all", "best"]:
            msg = f"The solutions parameter can be either 'all' or 'best' but {solutions} found."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if solutions == 'all':
            if self.save_solutions:
                # solutions_to_plot = np.array(self.solutions)
                # Group solutions by generation, it should probably be this way since the beginning
                solutions_to_plot = [
                    self.solutions[i * self.sol_per_pop:(i + 1) * self.sol_per_pop]
                    for i in range(self.generations_completed + 1)
                ]

                # Results in a matrix with shape (generations_completed + 1, sol_per_pop, num_genes)
                solutions_to_plot = np.array(solutions_to_plot)

            else:
                msg = "The plot_genes() method with solutions='all' can only be called if 'save_solutions=True' in the pygad.GA class constructor."
                self.logger.error(msg)
                raise RuntimeError(msg)
        elif solutions == 'best':
            if self.save_best_solutions:
                solutions_to_plot = self.best_solutions
            else:
                msg = "The plot_genes() method with solutions='best' can only be called if 'save_best_solutions=True' in the pygad.GA class constructor."
                self.logger.error(msg)
                raise RuntimeError(msg)

        if graph_type == "plot":

            # Validate inputs
            if figure_layout not in ["compact", "vertical_distribution", "horizontal_distribution"]:
                raise ValueError(f"Invalid figure format: {figure_layout}. Must be one of ['compact', 'vertical_distribution', 'horizontal_distribution']")

            if figure_layout == "compact":
                # num_rows will be always be >= 1
                # num_cols can only be 0 if num_genes=1
                num_rows = int(np.ceil(self.num_genes / 5.0))
                num_cols = int(np.ceil(self.num_genes / num_rows))

            elif figure_layout == 'vertical_distribution':
                num_rows = self.num_genes
                num_cols = 1

            elif figure_layout == 'horizontal_distribution':
                num_rows = 1
                num_cols = self.num_genes
    
            if num_cols == 0:
                figsize = (10, 8)
                # There is only a single gene
                fig, ax = plt.subplots(num_rows, figsize=figsize)
                if plot_type == "plot":
                    ax.plot(solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                elif plot_type == "scatter":
                    ax.scatter(range(self.generations_completed + 1), solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                elif plot_type == "bar":
                    ax.bar(range(self.generations_completed + 1), solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                ax.set_xlabel(0, fontsize=font_size)
            else:
                if axs is None:
                    fig, axs = plt.subplots(num_rows, num_cols)
    
                if num_cols == 1 and num_rows == 1:
                    fig.set_figwidth(5 * num_cols)
                    fig.set_figheight(4)
                    axs.plot(solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                    axs.set_xlabel("Gene " + str(0), fontsize=font_size)
                # elif num_cols == 1 or num_rows == 1:
                #     fig.set_figwidth(5 * num_cols)
                #     fig.set_figheight(4)
                #     for gene_idx in range(len(axs)):
                #         if plot_type == "plot":
                #             axs[gene_idx].plot(solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                #         elif plot_type == "scatter":
                #             axs[gene_idx].scatter(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                #         elif plot_type == "bar":
                #             axs[gene_idx].bar(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                #         axs[gene_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                else:
                    if axs is None:
                        fig.set_figwidth(10)
                        fig.set_figheight(3.5*num_rows)

                    # Ensure axs is a 2D array
                    axs = np.array(axs).reshape(num_rows, num_cols)

                    gene_idx = 0

                    for row_idx in range(num_rows):
                        for col_idx in range(num_cols):

                            if gene_idx >= self.num_genes:
                                # axs[row_idx, col_idx].remove()
                                break

                            # (generations_completed + 1, sol_per_pop, num_genes)
                            # One line for each gene and cromosome to show the individual evolution

                            args = (np.arange(0, self.generations_completed + 1), solutions_to_plot[:, :, gene_idx])
                            kwargs = {"linewidth": linewidth} #, "color": fill_color}

                            if plot_type == "plot":
                                axs[row_idx, col_idx].plot(*args, **kwargs)
                            elif plot_type == "scatter":
                                for trace_idx in range(solutions_to_plot.shape[1]):
                                    axs[row_idx, col_idx].scatter(args[0], args[1][:, trace_idx], **kwargs)

                            elif plot_type == "bar":
                                axs[row_idx, col_idx].bar(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)

                            axs[row_idx, col_idx].set_title(f"Gene {gene_idx if self.gene_names is None else self.gene_names[gene_idx]}", fontsize=font_size)
                            if col_idx == 0:
                                axs[row_idx, col_idx].set_ylabel("Values", fontsize=font_size)
                            if row_idx == num_rows - 1:
                                axs[row_idx, col_idx].set_xlabel("Generations", fontsize=font_size)

                            gene_idx += 1

            if axs is None:
                fig.suptitle(title, fontsize=font_size, y=1.001)
                plt.tight_layout()

        elif graph_type == "boxplot":
            fig = plt.figure(1, figsize=(0.7*self.num_genes, 6))

            # Create an axes instance
            ax = fig.add_subplot(111)
            boxeplots = ax.boxplot(solutions_to_plot, 
                                   labels=range(self.num_genes),
                                   patch_artist=True)
            # adding horizontal grid lines
            ax.yaxis.grid(True)
    
            for box in boxeplots['boxes']:
                # change outline color
                box.set(color='black', linewidth=linewidth)
                # change fill color https://color.adobe.com/create/color-wheel
                box.set_facecolor(fill_color)

            for whisker in boxeplots['whiskers']:
                whisker.set(color=color, linewidth=linewidth)
            for median in boxeplots['medians']:
                median.set(color=color, linewidth=linewidth)
            for cap in boxeplots['caps']:
                cap.set(color=color, linewidth=linewidth)
    
            plt.title(title, fontsize=font_size)
            plt.xlabel(xlabel, fontsize=font_size)
            plt.ylabel(ylabel, fontsize=font_size)
            plt.tight_layout()

        elif graph_type == "histogram":
            # num_rows will always be >= 1
            # num_cols can only be 0 if num_genes=1
            num_rows = int(np.ceil(self.num_genes/5.0))
            num_cols = int(np.ceil(self.num_genes/num_rows))
    
            if num_cols == 0:
                figsize = (10, 8)
                # There is only a single gene
                fig, ax = plt.subplots(num_rows,
                                                     figsize=figsize)
                ax.hist(solutions_to_plot[:, 0], color=fill_color)
                ax.set_xlabel(0, fontsize=font_size)
            else:
                fig, axs = plt.subplots(num_rows, num_cols)
    
                if num_cols == 1 and num_rows == 1:
                    fig.set_figwidth(4 * num_cols)
                    fig.set_figheight(3)
                    axs.hist(solutions_to_plot[:, 0], 
                             color=fill_color,
                             rwidth=0.95)
                    axs.set_xlabel("Gene " + str(0), fontsize=font_size)
                elif num_cols == 1 or num_rows == 1:
                    fig.set_figwidth(4 * num_cols)
                    fig.set_figheight(3)
                    for gene_idx in range(len(axs)):
                        axs[gene_idx].hist(solutions_to_plot[:, gene_idx], 
                                           color=fill_color,
                                           rwidth=0.95)
                        axs[gene_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                else:
                    gene_idx = 0
                    fig.set_figwidth(20)
                    fig.set_figheight(3*num_rows)
                    for row_idx in range(num_rows):
                        for col_idx in range(num_cols):
                            if gene_idx >= self.num_genes:
                                # axs[row_idx, col_idx].remove()
                                break
                            axs[row_idx, col_idx].hist(solutions_to_plot[:, gene_idx], 
                                                       color=fill_color,
                                                       rwidth=0.95)
                            axs[row_idx, col_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                            gene_idx += 1
    
            fig.suptitle(title, fontsize=font_size, y=1.001)
            plt.tight_layout()

        if save_dir is not None:
            plt.savefig(fname=save_dir, bbox_inches='tight')

        if show_plot:
            plt.show()

        return fig if axs is None else axs[0,0].figure

    def grouped_plot(self,
                     show_plot: bool = True,
                     figure_layout: Literal['vertical', 'horizontal'] = 'horizontal',
                     figsize: tuple[float, float] = None,
                     save_dir: str | Path = None) -> plt.Figure:

        # Set figure layout
        if figure_layout == 'vertical':
            n_rows = 2 + self.num_genes
            n_cols = 1

            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7, 2 * n_rows) if not figsize else figsize, sharex=True)

        elif figure_layout == 'horizontal':
            n_rows = 2
            n_cols = np.max([2, self.num_genes])

            gs = gridspec.GridSpec(n_rows, n_cols, )
            fig = plt.figure(figsize=(2.5 * n_cols, 7) if not figsize else figsize)


        for idx, plot_type in enumerate(['fitness', 'new_solution_rate', 'genes']):
            xlabel = None if idx == 0 and figure_layout == 'vertical' else "Generation"

            if figure_layout == 'vertical':
                ax = axs[idx] if plot_type != "genes" else axs[-self.num_genes:]
                figure_layout = "vertical_distribution"

            elif figure_layout == 'horizontal':
                if plot_type != "genes":
                    if idx == 0:
                        ax = fig.add_subplot(gs[0, :n_cols//2])
                    else:
                        ax = fig.add_subplot(gs[0, n_cols//2:])
                else:
                    ax = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
                    figure_layout = "horizontal_distribution"

            if plot_type == "fitness":
                self.plot_fitness(ax=ax, show_plot=False, xlabel=xlabel, )
            elif plot_type == "new_solution_rate":
                self.plot_new_solution_rate(ax=ax, show_plot=False, xlabel=xlabel, )
            elif plot_type == "genes":
                self.plot_genes(axs=ax, show_plot=False, plot_type="scatter", graph_type="plot", figure_layout=figure_layout)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

        fig.suptitle("PyGAD - Optimization results")
        plt.tight_layout()

        if save_dir is not None:
            self.save_figure(save_dir)

        if show_plot:
            plt.show()

        return fig

