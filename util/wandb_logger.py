import os
import wandb

import util.logger as logger

class WandbLogger(logger.Logger):
    MISC_TAG = "Misc"

    def __init__(self, project_name, exp_name, connect_to_wandb, param_config=None):
        super().__init__()
        
        self._project_name = project_name
        self._connect_to_wandb = connect_to_wandb
        self._exp_name = exp_name
        self._param_config = param_config
        self._step_var_key = None
        self._collections = dict()

        return
    
    def reset(self):
        super().reset()
        return

    def configure_output_file(self, filename=None):
        super().configure_output_file(filename)

        if (logger.Logger.is_root() and self._connect_to_wandb):
            wandb.init(project=self._project_name, name=self._exp_name, config=self._param_config)
        
        return

    def set_step_key(self, var_key):
        self._step_key = var_key
        return

    def log(self, key, val, collection=None, quiet=False):
        super().log(key, val, quiet)

        if (collection is not None):
            self._add_collection(collection, key)
        return

    def write_log(self):
        row_count = self._row_count

        super().write_log()

        if (logger.Logger.is_root() and (wandb.run is not None)):
            if (row_count == 0):
                self._key_tags = self._build_key_tags()
            
            step_val = row_count
            if (self._step_key is not None):
                step_val = self.log_current_row.get(self._step_key, "").val
            

            out_dict = dict()
            for i, key in enumerate(self.log_headers):
                if (key != self._step_key):
                    entry = self.log_current_row.get(key, "")
                    val = entry.val
                    tag = self._key_tags[i]
                    out_dict[tag] = val

            if self._connect_to_wandb:
                wandb.log(out_dict, step=int(step_val))

        return
    
    def _add_collection(self, name, key):
        if (name not in self._collections):
            self._collections[name] = []
        self._collections[name].append(key)
        return
    
    def _build_key_tags(self):
        tags = []
        for key in self.log_headers:
            curr_tag = WandbLogger.MISC_TAG
            for col_tag, col_keys in self._collections.items():
                if key in col_keys:
                    curr_tag = col_tag

            curr_tags = "{:s}/{:s}".format(curr_tag, key)
            tags.append(curr_tags)

        return tags
    
    def end_wandb(self):
        wandb.finish()
        return