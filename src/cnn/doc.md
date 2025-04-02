## Supported models by Optium:

#### Intel:
- Includes Llama, GPT, Bert, DeepSeek, Gemma, Mistral, Qwen
- https://huggingface.co/docs/optimum/intel/openvino/models

#### AMD:
- Supports Ryen AI
- Not compatible with transformers pipeline
- Apparently TIMM is a extensions, that can enable transformers
- Support limited to image classification, object-detection and image segmentation
- 

## Runttime environment
Dependencies:
- Python=3.12
- transformers>=4.50
- optimum
- (transformers[sentencepiece])
- nncf
- optimum[intel]
- (optimum-nvidia (pin install --extra-index-url https://pypi.nvidia.com optimum-nvidia))
- 

#### Conifgs:
- .wslconfig set to use 20 Gb RAM for 32 Gb computer to run Llama-3B on huggingface