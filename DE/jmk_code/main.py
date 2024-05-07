import utils
import cano

if __name__=='__main__':
  ACCESS_TOKEN = utils.get_access_token()
  utils.ACCESS_TOKEN = ACCESS_TOKEN
  cano.ACCESS_TOKEN = ACCESS_TOKEN

  symbol_list = ['005930','035720','000660','069500']

  utils.send_message('=== 프로그램 시작 ===')
  current_price = utils.get_current_price(symbol_list[0])
  print(current_price)