import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch


class AdversarialEvaluation:

    """
    Performs evaluation to adversarial attacks
    """

    def __init__(self, train_loader, test_loader, model, loss, optimizer, device, arguments, **kwargs):
        self.args = arguments
        self.fb = fb
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.model = model
        self.loader_test = test_loader
        self.loader = train_loader

    def evaluate(self, plot=False, targeted=False, exclude_wrong_predictions=False):

        method = self.args.attack

        model = self.model.eval().to(self.device)
        im, crit = next(iter(self.loader_test))
        bounds = (im.min().item(), im.max().item())
        fmodel = self.fb.PyTorchModel(model, bounds=bounds, device=self.device)

        im = im.to(self.device)
        crit = crit.to(self.device)
        probs = model.forward(im)
        predictions = probs.argmax(dim=-1)

        if exclude_wrong_predictions:
            selection = predictions == crit
            im = im[selection]
            crit = crit[selection]
            predictions = predictions[selection]

        if targeted:
            target = 1
            selection = crit != target
            im = im[selection]
            predictions = predictions[selection]
            miss_classifications = torch.tensor([target] * len(im))
            crit = self.fb.criteria.TargetedMisclassification(
                miss_classifications)

        attack = self.get_attack(method)

        plt.title(str(attack.__class__).split(".")[-1])

        sucess_rates = []

        epsilons = [0.25, 0.5, 0.75, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        _, advs, success = attack(fmodel, im, crit, epsilons=epsilons)
        for eps, eps_adv, eps_success in zip(epsilons, advs, success):
            attack_success = (eps_success.float().sum() / len(eps_success)).item()
            adv_equality = (model.forward(eps_adv).argmax(dim=-1) == predictions)
            predicted_same_as_model = (adv_equality.float().sum() / len(eps_success)).item()

            sucess_rates.append(attack_success)

            print("EPSILON", eps,
                  "Successes attack", attack_success,
                  "same prediction as model", predicted_same_as_model,
                  "bounds adver", eps_adv.min().item(), eps_adv.max().item(),
                  "norm", np.mean([torch.norm(eps_adv_ - im_, p=2).item() for eps_adv_, im_ in zip(eps_adv, im)]))

            if plot:
                self._plot(adv_equality, eps, eps_adv, im, model)

        return epsilons, np.array(sucess_rates)

    def get_attack(self, method_name):

        att = self.fb.attacks

        print("> PERFORMING ADV ATTACK", method_name)

        switcher = {
            "CarliniWagner": att.L2CarliniWagnerAttack,
            "LinfPGD": att.LinfPGD,
            "L1FastGradientAttack": att.L1FastGradientAttack,
            "L2DeepFoolAttack": att.L2DeepFoolAttack,
            "FGSM": att.FGSM,
            "DDNAttack": att.DDNAttack,
            "SaltAndPepperNoiseAttack": att.SaltAndPepperNoiseAttack,
            "L2RepeatedAdditiveGaussianNoiseAttack": att.L2RepeatedAdditiveGaussianNoiseAttack,
        }
        attack = switcher.get(method_name, f"{method_name} not recognised")()
        return attack
