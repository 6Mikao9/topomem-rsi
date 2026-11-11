# North Wharf Archive incident appendix

This supplementary record belongs to the same fictional archive exercise. It gives the incoming shift enough detail to inspect the evidence without using a public notice as an engineering log. Times refer to the morning in the main briefing. Materials described as examples are examples from the validation exercise. A planned action remains a plan until the responsible operator records a receipt or the incident commander records a decision.

## How a date enters the catalog

The original issue arrives as a scan, an edition label supplied by the catalog desk, and a source checksum. The raw printed date stays attached to the edition even when the normalized date is empty. Ingestion can accept a file whose date is ambiguous because archival preservation and searchable date facets serve different purposes. An edition with an empty normalized date remains reachable by title and edition identifier. It might be absent from a search restricted to a particular calendar day. This distinction matters when support explains why two displays looked different. A missing day in a filter and a missing original file call for different follow-up.

The catalog desk records locale evidence separately from parser output. A publication style guide can specify a date order for a known run of editions. A complete month name can identify the intended month without relying on a locale. A region code on the website identifies the service location that returned an answer. That code never supplies evidence about where a newspaper was printed or how its editor wrote dates. The experiment tests whether a parser respects these separate roles. It should continue to accept confidently dated editions while keeping an unresolved slash date available for later catalog work.

## Examples for the validation meeting

One validation record contains the printed form `13/04` and a style guide that expressly uses day first. The reviewer can check April 13 against both forms of evidence. A second contains `April 13` and needs no numerical order decision. A third contains `04/05` with two possible orders and carries an import locale that describes the workstation used for digitization. That workstation setting does not resolve the printed date. Its expected normalized field is empty. A fourth record has an explicit editorial day-first annotation but a catalog transfer sheet giving month first. The conflict should stay visible for review, and the search index should leave its normalized value blank.

Several of the twelve examples share a numerical form while varying their supporting evidence. A correct parser can therefore return different answers for records whose printed date strings happen to be identical. The candidate validation report should show the raw string, evidence category, previous normalized value and candidate value for each example. The reviewer should be able to see when a new rule leaves a date blank on purpose. A row that passes because a fallback guessed the reviewer’s expected answer still needs an explanation. It does not establish that the fallback will be reliable on another edition.

## What the two replay requests establish

The unambiguous control, R17, checks that both generations can produce the same expected answer when date order is irrelevant. It offers a useful control for the replay method because an accidental change to the query or index might also affect an ordinary case. R18 selects the ambiguous edition and preserves the full incoming request used for both comparisons. During the limited trial the controlled result changes with the parser generation. It stays the same for repeated invocations against a given generation. The local invocation that bypasses the public proxy preserves the G42 result. These observations establish a specific mechanism for the reproduced case.

The replay cannot establish an incident-wide count. A date field can differ in a catalog record that nobody queried during the trial, and one reader can make several queries. A reader can also report two results for the same edition and count it as a single issue. The fixed replay requests make it possible to isolate the cause without inferring a readership denominator. The follow-up sample consists of six reports received through support. Four fit the slash-date pattern. The two named-month reports remain open because their symptoms did not reproduce after the rollback. The operations handover can mention that split when it helps the next shift keep the cases separate.

## Distinguishing checksums, requests and generations

The immutable checksum refers to the archived source object. Agreement on that checksum shows the two displays point to the same underlying object for the observed case. A checksum does not guarantee agreement on all catalog metadata. A committed generation identifies the searchable view. The generation can change while the checksum stays stable because the parser affects a derived field. The replay request bytes identify the input presented to the serving path. Keeping them fixed removes an otherwise plausible explanation that the two clients asked subtly different questions.

The diagnostic region header is still useful for tracing where a response was served. It simply has a different role from these other identifiers. When recording an observation, put the generation and region alongside the document identifier and source checksum. Keep the replay request code in the engineering attachment. The public update can describe the observation in ordinary language. A reader does not need an internal request identifier to understand why the archive returned to the prior date handling.

## Containment receipt and its boundaries

At 09:20 the incident commander directed that G42 promotion stop. The deployment controller recorded the east return to G41 at 09:21. That receipt is the last confirmed change to the public reader state. The west group had stayed on G41 throughout. After the receipt, the controlled comparison again agreed across regions and showed the disputed field blank. That result is consistent with the old parser behavior. It does not require repair of an original scan, replay of the ingestion queue or a replacement of the catalog document identifier.

Keep the stopped promotion separate from the rollback receipt. The former is an operational instruction about future exposure. The latter shows which generation is actually serving. A reader who saw the G42 result earlier should be told that the archive restored the previous handling. The team should avoid describing every historical edition as corrected because the older handling deliberately leaves some dates empty. Containment ends the known exposure to the candidate parser while validation work continues.

## How to use the monitoring board

The board tracks request errors, search latency and queue depth. The relevant observations show normal error rate and an ordinary morning queue after containment. This helps rule out an accompanying service-wide disruption. A date interpretation difference can still exist with healthy response times and no increase in failures. An operator investigating a future catalog mismatch should start with the response generation and the document record, even when the service dashboard looks normal.

An ingestion queue graph answers a different question from the generation replay. It can show when indexing work falls behind or when new editions take longer to appear. It cannot, by itself, show why two reads of the same committed edition label produced different derived dates. A long queue might matter in another incident. For this handover, the controlled G42 replay and signed rollback receipt carry more evidentiary weight than the queue remaining near its expected size.

## What qualifies as locale evidence

Use the publisher’s edition-specific guide where one exists. A source transcription may include a month name or an explicit language convention tied to the printed edition. An earlier catalog entry can help a human reviewer find supporting evidence but should not silently become the only reason a parser picks a date. Publication place is a weak cue when the edition uses a different editorial convention. Digitization site, ingestion host, reader location and browser language describe the process around the document. None directly indicates the intended order of ambiguous numbers in its masthead.

When several sources agree, record the strongest direct one and enough context to reproduce the decision. When they conflict, preserve the disagreement and send it to catalog review. An empty date can be a deliberate outcome of a sound validation rule. That outcome should have a recorded reason so the next reviewer can distinguish a known ambiguity from an ingestion error or a missing scan.

## Shift desk decision table

An ordinary complaint with no generation information starts as a request for the edition identifier and the time the problem was observed. The help desk can collect diagnostic headers if the reader provides them. It should avoid asking for account details. A report that explicitly shows G42 during the east trial can be compared with the confirmed reproduction. A report that explicitly shows G41 after rollback should go into the new investigation queue, with a link to the older incident for context and a clear note that its cause remains open.

If the deployment view unexpectedly shows G42 before the checkpoint, the on-call operator should confirm the receipt and inform the incident commander rather than treating the candidate tests as authorization. If both regions show G41 and the only new reports involve the previous trial interval, leave the generation in place and keep triaging reports. If the validation meeting is delayed, update the checkpoint as a pending review. Do not substitute an estimate for an actual decision to start another trial.

## Validation artifacts to prepare

Before 11:15, the ingestion lead should assemble the twelve-row matrix, the evidence excerpts approved for engineering review, and the candidate parser's output for each row. The serving lead can provide the signed deployment and rollback receipts plus the controlled replay log. The incident commander needs a concise note on what changed, what remains in service, which outcomes are still unknown and who will make the next decision. The artifacts can be ready before a release decision exists.

Keep the candidate configuration beside the validation report so later changes do not appear to be covered by the same result. A reviewer should be able to identify whether a test exercised the exact ambiguous branch discussed at the checkpoint. A successful test result would justify considering another limited trial. It would not by itself change either region. A change in production requires a separate authorization, an explicit scope and a newly observed deployment receipt.

## Notes for the public help desk

An update can explain that the archive restored its earlier handling of edition dates after a limited catalog trial produced inconsistent displays. It can also say that the original materials were unchanged and that date-handling rules are being reviewed. That message answers the reasonable question a reader may have about the newspaper itself. Keep the wording short enough for the support template and avoid naming the operators or engineering request labels.

Do not promise that the disputed date will appear on a particular day. The prior policy intentionally leaves unresolved dates blank. The checkpoint is an internal review and does not promise a new trial. Avoid phrases that imply a complete count of readers or editions, because the sample was collected for follow-up rather than population measurement. If readers ask how to find an edition whose day is blank, the help desk can explain that the title and edition identifier still identify the archived object.

## Preserving tentative explanations

The first incident thread included a suggestion that a public network replay might have modified the request. It was a plausible hypothesis when the symptom first appeared in two different regions. The controlled comparison later made it unnecessary to explain the reproduced case: the discrepancy followed G42 even when the public proxy was bypassed. Preserve the hypothesis in the engineering history if useful, but attach its later disposition. The handover should not list it alongside the confirmed cause as an equally live explanation for that case.

The two unreproduced named-month reports can have separate causes. Do not close them by saying the generation replay proved all catalog differences originated in the same rule. A new investigation can look for other reproducible inputs, compare recorded generation headers and establish its own controls. If it finds a distinct condition, its observations should be recorded independently of the limited slash-date trial.

## Handover readiness checklist

The incoming operations engineer should be able to identify G41 as active in both reader groups, locate the 09:21 receipt and state the purpose of the 11:15 review. They should know that request bytes and the source checksum stayed fixed in the controlled replay, that the ambiguous result followed G42 and that a proxy bypass reproduced it. They should also know which cases remain outside that conclusion. If the handover includes a recommendation, name the supporting evidence and the person responsible for approving any subsequent change.

For the incoming help desk operator, the essential facts are that earlier date handling was restored after a limited trial, the original files were unchanged, and the archive is reviewing how to handle ambiguous date formats. Their message should not read like a deployment log. The public and operations handovers can differ in detail while remaining consistent about the catalog field, the rollback and the pending review. Record uncertainty plainly where a reader might otherwise infer more than the exercise establishes.

## An explicit stopping point

The current record ends before the 11:15 decision. It contains no candidate test outcome, no subsequent trial and no broader incident count. When a request asks what the next checkpoint means, use the validation and review described here. When a request asks what the archive has already done, use the signed rollback and observed return to G41. The same distinction applies to all exercises that use this document. Do not add a plausible final chapter when the task is to summarize the documented morning.
