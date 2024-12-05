'A module for initializing complex data types for hints'

from typing import Tuple, Optional

import torch


dataset_item_type = Tuple[
                            Tuple[                      # Batch tuple
                                int,                        # user entity id
                                int,                        # item entity id
                                Optional[torch.Tensor],     # user features tensor
                                Optional[torch.Tensor]],    # item features tensor
                            torch.Tensor                # Labels tuple
                        ]