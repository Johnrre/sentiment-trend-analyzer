import time
import logging
import schedule

from news_pull import run_daily_pull


handler = [logging.StreamHandler(), logging.FileHandler("./data/scheduler.log")]
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=handler)

log = logging.getLogger(__name__)

def pull():
  log.info("Starting scheduled news pull....")
  try:
    count = run_daily_pull()
    log.info(f"Pull successfull. {count} articles added!")
  except Exception as e:
    log.info(f"Pull failed. {e}", exc_info=True)

schedule.every().day.at("08:00").do(pull)
schedule.every().day.at("20:00").do(pull)

if __name__ == "__main__":
    try:
      log.info("Scheduler started. Waiting for next job...")
      log.info(f"Next run: {schedule.next_run()}\n")
  
      while True:
          schedule.run_pending()
          time.sleep(60)

    except KeyboardInterrupt:
       print("...program has stopped!")
  
