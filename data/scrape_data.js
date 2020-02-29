const { HLTV } = require('hltv')

HLTV.getMatch({id: 2306295}).then(res => {
    console.log(res);
})