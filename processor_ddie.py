from transformers.data.processors.utils import InputExample, InputFeatures, DataProcessor
import os,logging

logger = logging.getLogger(__name__)
class DDIProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(tensor_dict["idx"].numpy(),
                            tensor_dict["sentence1"].numpy().decode("utf-8"),
                            tensor_dict["sentence2"].numpy().decode("utf-8"),
                            str(tensor_dict["label"].numpy()))

    def get_train_examples(self, data_dir):
        logger.info("looking at {}".format(os.path.join(data_dir, "train.tsv"))) 
        return self._create_example(
                self._read_tsv(os.path.join(data_dir,"train.tsv")),"train")
    
    def get_dev_examples(self, data_dir):
        return self._create_example(
                self._read_tsv(os.path.join(data_dir,"dev.tsv")),"dev")
    
    def get_examples(self, data_dir):
        return self._create_example(
            self._read_tsv(os.path.join(data_dir,"total.tsv")),"total")
    
    def get_test_examples(self, data_dir):
        return self._create_example(
                self._read_tsv(os.path.join(data_dir,"test.tsv")),"test")

    def get_labels(self, mode = "ddie"):
        if mode == "ddie":
            #return ['negative', 'mechanism', 'effect', 'advise', 'int']
            return ['other', 'mechanism', 'effect', 'advise', 'int']
        if mode == "pretraining":
            return ['negative', 'positive']

    def _create_example(self, lines, set_type):
        examples = []
        for i,line in enumerate(lines):
            guid = "%s-%s" % (set_type, i) # trian-1;trian-2;......////// dev-1;dev-2;......
            text_a = line[1] # text
            label = line[0] # negative
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label) 
            )
        return examples
        
class DescProcessor(DDIProcessor):
    def get_train_examples(self, data_dir, drug_index):
        logger.info("looking at {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_example(
            self._read_tsv(os.path.join(data_dir,"train.tsv")),"train", drug_index)
    
    def get_dev_examples(self, data_dir, drug_index):
        return self._create_example(
            self._read_tsv(os.path.join(data_dir,"dev.tsv")), "dev", drug_index)
    
    def get_examples(self, data_dir, drug_index):
        return self._create_example(
            self._read_tsv(os.path.join(data_dir,"total.tsv")), "total", drug_index)
    
    def get_test_examples(self, data_dir, drug_index):
        return self._create_example(
                self._read_tsv(os.path.join(data_dir,"test.tsv")),"test", drug_index)

    def _create_example(self, lines, set_type, drug_index):
        examples = []
        for (i,line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if drug_index == 1:
                text_a = line[4] 
            if drug_index == 2:
                text_a = line[5] 
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label) 
                # examples:[InputExample1,InputExample2,InputExample3,...]
                # InputExamle.guid = dev-0
                # InputExamle.text_a = 'The administration of quinidine derivatives helps to observe various skin and mucosal reactions. A papulopurpuric eruption in a  patient (without thrombopenia) can be developed who is taking quinidine phenylethyl barbiturate intermittently and at reintroduction.(PMID: 9739909)',
                # InputExamle.text_b = ""
                # InputExamle.label = negative
            )
        return examples
        
ddie_processors = {"mrpc": DDIProcessor, "desc": DescProcessor}
