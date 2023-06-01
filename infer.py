import argparse
import logging

if __name__ == "main":
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
  parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
  parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
  parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')

  args = parser.parse_args()
#   opt = Logger.parse(args)