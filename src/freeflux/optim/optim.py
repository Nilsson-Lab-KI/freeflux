"""Define the Optimizer class."""

__author__ = 'Chao Wu'
__date__ = '04/14/2022'

from functools import partial
from collections.abc import Iterable
from ..solver.lpsolver import FBAModel
from ..io.results import FBAResults, FVAResults
from ..utils.progress import Progress
from ..utils.context import Context
from ..core.model import Model


class Optimizer:
    """
    Parameters
    ----------
    model: Model
        Freeflux Model.
    """

    model: Model

    def __init__(self, model: Model):
        """
        Parameters
        ----------
        model: Model
            Freeflux Model.
        """

        self.model = model
        self.contexts = []

    def __enter__(self):
        self.contexts.append(Context())
        return self

    def __exit__(self, type, value, traceback):
        context = self.contexts.pop()
        context.undo()

    def _set_flux_bounds(self, flux_id: str, bounds):
        """
        Parameters
        ----------
        flux_id: str
            Flux ID, i.e. reaction ID. For irreversible reaction, the lower bound of range will
            be set to zero ignorant of bounds[0].
        bounds: 2-list
            [lower bound, upper bound].
        """
        bounds = list(map(float, bounds))
        if self.model.reactions_info[flux_id].reversible:
            self.model.net_fluxes_bounds[flux_id] = list(bounds)
        else:
            if bounds[1] >= 0.0:
                self.model.net_fluxes_bounds[flux_id] = [max(0.0, bounds[0]), bounds[1]]
            else:
                raise ValueError(f'flux bounds of {flux_id} conflict with its reversibility')

    def set_flux_bounds(self, flux_id, bounds):
        """
        Set lower and upper bounds for metabolic flux. If called several times,
        the latest setting overrides previous ones if conflicted.

        Parameters
        ----------
        flux_id: str or 'all'
            Flux ID, i.e. reaction ID. This method is used to set the range of net flux.
            For irreversible reaction, the lower bound will be set to zero ignorant of bounds[0].

            If 'all', all fluxes will be set to the range.
        bounds: 2-list
            [lower bound, upper bound]. Let lower bound equals upper bound to set fixed value.
        """
        flux_ids = []
        if bounds[0] <= bounds[1]:
            if flux_id == 'all':
                for flux_id in self.model.netfluxids:
                    self._set_flux_bounds(flux_id, bounds)
                    flux_ids.append(flux_id)
            elif flux_id in self.model.netfluxids:
                self._set_flux_bounds(flux_id, bounds)
                flux_ids = [flux_id]
            else:
                raise ValueError(f'flux range set to nonexistent reaction {flux_id}')
        else:
            raise ValueError('lower bound of flux should not be greater than upper bound')

        if self.contexts:
            context = self.contexts[-1]
            context.add_undo(partial(self._unset_flux_bounds, flux_ids))

    def _set_default_flux_bounds(self):
        """
        This method assigns bounds of [-100, 100] for reversible reactions and
        [0, 100] for irreversible reactions not yet set by set_flux_bounds
        """
        default_bounds = [-100, 100]
        flux_ids = []
        for flux_id in self.model.netfluxids:
            if flux_id not in self.model.net_fluxes_bounds:
                self._set_flux_bounds(flux_id, default_bounds)
                flux_ids.append(flux_id)

        if self.contexts:
            context = self.contexts[-1]
            context.add_undo(partial(self._unset_flux_bounds, flux_ids))

    def _unset_flux_bounds(self, flux_ids):
        """
        Parameters
        ----------
        flux_ids: str or list of str
            Flux ID(s).
        """
        if not isinstance(flux_ids, Iterable):
            flux_ids = [flux_ids]
        for flux_id in flux_ids:
            if flux_id in self.model.net_fluxes_bounds:
                self.model.net_fluxes_bounds.pop(flux_id)

    def prepare(self):
        """
        This method do some preparation work.
        """

        self._set_default_flux_bounds()

    def _check_stoichiometric_matrix(self, exclude_metabs):
        """
        Check and delete zero rows or columns usually caused by dilution reactions.

        Parameters
        ----------
        exclude_metabs: list
            Metabolite IDs, i.e., metabolites excluded from mass balance.
        """

        netfluxids = self.model.netfluxids.copy()

        net_fluxes_bounds = self.model.net_fluxes_bounds.copy()

        stoy_mat = self.model.get_net_stoichiometric_matrix(exclude_metabs).copy(deep='all')

        maskZeroRows = (stoy_mat.T != 0.0).any()
        stoy_mat = stoy_mat[maskZeroRows]

        maskZeroCols = (stoy_mat != 0.0).any()
        netfluxids_todrop = stoy_mat.columns[~maskZeroCols].tolist()

        if netfluxids_todrop:
            for fluxid in netfluxids_todrop:
                print(
                    f'{fluxid} not connected with other reactions, '
                    f'removed from flux balance'
                )

                if fluxid in netfluxids:
                    netfluxids.remove(fluxid)
                if fluxid in net_fluxes_bounds:
                    net_fluxes_bounds.pop(fluxid)

            stoy_mat = stoy_mat.T[maskZeroCols].T

        return netfluxids, net_fluxes_bounds, stoy_mat

    def optimize(
            self,
            objective,
            direction='max',
            exclude_metabs=None,
            show_progress=True
    ):
        """
        This method performs flux balance analysis by maximizing the objective.

        Parameters
        ----------
        objective: dict
            Reaction ID => coefficient, i.e., objective function.
        direction: {"max", "min"}
            Optimization direction.
        exclude_metabs: list
            Metabolite IDs, i.e., metabolites excluded from mass balance.
        show_progress: bool
            Whether to show the progress bar.

        Returns
        -------
        FBAResults: FBAResults
        """

        (netfluxids,
         net_fluxes_bounds,
         stoy_mat
         ) = self._check_stoichiometric_matrix(exclude_metabs)

        fbaModel = FBAModel()
        fbaModel.build_flux_variables(netfluxids, net_fluxes_bounds)
        fbaModel.build_objective(objective, direction)
        fbaModel.build_mass_balance_constraints(stoy_mat)

        with Progress('optimizing', silent=not show_progress):
            optObj, optFluxes = fbaModel.solve_flux()

        return FBAResults(objective, optObj, optFluxes)

    def estimate_fluxes_range(
            self,
            objective=None,
            gamma=0,
            exclude_metabs=None,
            show_progress=True
    ):
        """
        This method performs flux variability analysis and estimates the ranges of metabolic fluxes.
        If an objective is provided, it will be optimized in maximizing direction.

        Parameters
        ----------
        objective: dict
            Reaction ID => coefficient, i.e., objective function.
        gamma: float
            During evaluation of each flux, objective value is required to be
            >= gamma*max(objective) with gamma in [0, 1]
        exclude_metabs: list
            Metabolite IDs, i.e., metabolites excluded from mass balance.
        show_progress: bool
            Whether to show the progress bar.

        Returns
        -------
        FVAResults: FVAResults
        """

        (netfluxids,
         net_fluxes_bounds,
         stoy_mat
         ) = self._check_stoichiometric_matrix(exclude_metabs)

        fvaModel = FBAModel()
        fvaModel.build_flux_variables(netfluxids, net_fluxes_bounds)
        fvaModel.build_mass_balance_constraints(stoy_mat)
        if objective is not None and gamma != 0:
            fvaModel.build_objective(objective, 'max')
            optObj, _ = fvaModel.solve_flux()
            fvaModel.remove_objective()
            fvaModel.build_objective_constraint(objective, optObj, gamma)

        with Progress('estimating flux ranges', silent=not show_progress):

            fluxRanges = {}
            for fluxid in self.model.netfluxids:
                if fluxid in netfluxids:
                    objective = {fluxid: 1}

                    fvaModel.build_objective(objective, 'min')
                    fluxLB, _ = fvaModel.solve_flux()
                    fvaModel.remove_objective()

                    fvaModel.build_objective(objective, 'max')
                    fluxUB, _ = fvaModel.solve_flux()
                    fvaModel.remove_objective()
                else:
                    fluxLB, fluxUB = self.model.net_fluxes_bounds[fluxid]

                fluxRanges[fluxid] = [fluxLB, fluxUB]

        return FVAResults(fluxRanges)
