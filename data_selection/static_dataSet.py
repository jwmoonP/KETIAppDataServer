
def set_integratedDataInfo(start, end):
    intDataInfo = {
            "Data":[{"db_name":"INNER_AIR", "measurement":"HS1", "domain":"farm", "subdomain":"airQuality"},
                    {"db_name":"OUTDOOR_AIR", "measurement":"sangju", "domain":"city", "subdomain":"airQuality" },
                    {"db_name":"OUTDOOR_WEATHER", "measurement":"sangju", "domain":"city", "subdomain":"weather"}],
            "start":str(start),
            "end":str(end)
    }
    return intDataInfo